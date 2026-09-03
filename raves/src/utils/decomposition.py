import warnings
import numpy as np
from typing import Tuple

from scipy.sparse import coo_array, csr_array, lil_array, dia_array, block_array
from scipy.sparse.linalg import ArpackNoConvergence, eigs


def eig_to_T60(eigenvalue: float, fs: float) -> float:
    """
    Translate eigenvalue magnitude to T60 (seconds).

    If abs(eigenvalue) >= 1, the decay does not converge and T60 is set to
    infinity. If abs(eigenvalue) == 0, T60 is 0. Otherwise the mapping
    uses the base-10 logarithm:

        T60 = -6 / (log10(abs(eigenvalue)) * fs)

    Parameters
    ----------
    eigenvalue : float
        Real eigenvalue (pole) of the state transition matrix.
    fs : float
        Sample rate used in the decomposed ART model, in Hz.

    Returns
    -------
    float
        Reverberation time in seconds.
    """
    if np.abs(eigenvalue) >= 1:
        return np.inf
    elif np.abs(eigenvalue) == 0:
        return 0.
    else:
        return -6 / (np.log10(np.abs(eigenvalue)) * fs)


def T60_to_eig(T60: float, fs: float) -> float:
    """
    Translate T60 (seconds) to eigenvalue magnitude.

    If T60 is 0, the result is 0. For finite nonzero T60, it is

        eig = 10 ** (-6 / (T60 * fs))

    and for non-finite T60 the result is 1.

    Parameters
    ----------
    T60  : float
        Reverberation time in seconds.
    fs : float
        Sample rate used in the decomposed ART model, in Hz.

    Returns
    -------
    float
        Real eigenvalue magnitude in [0, 1].
    """
    if T60 == 0:
        return 0.
    elif np.isfinite(T60):
        return 10**(-6 / (T60 * fs))
    else:
        return 1.



def build_ssm(kernel: csr_array, m: np.ndarray,
              element_wise_assembly: bool = True
              ) -> csr_array:
    """
    Construct the sparse state transition matrix for an FDN-like system.

    Given an N x N feedback matrix and N integer delay lengths `m` (each
    at least 3), this builds the state transition matrix (size sum(m) x sum(m))
    that advances all inner states by one sample and feeds the last samples
    through the feedback matrix and then into the first samples.

    Two assembly modes are supported:
    - element_wise_assembly=True: set individual nonzeros directly in a LIL
      matrix and convert to CSR (fast and memory-friendly for large N).
    - element_wise_assembly=False: assemble using block matrices:
        * U_i: shifts inner samples of line i (shape (m_i-2, m_i-2))
        * R_i: shifts the first sample to the second (shape (m_i-2, N))
        * P_i: shifts the last sample into the first (shape (N, m_i-2))
        * feedback matrix: couples last to first samples across lines
    The latter assembly mode is included to intuitively match the definition
    shown in our papers.

    Parameters
    ----------
    kernel : scipy.sparse.csr_array
        Feedback matrix of shape (N, N), where N == len(m).
    m : array_like of int
        Per-line integer delay lengths; each m_i must be >= 3.
    element_wise_assembly : bool, default True
        If True, assemble by writing individual entries.
        If False, assemble from sparse blocks.

    Returns
    -------
    csr_array
        Sparse state transition matrix of shape (sum(m), sum(m)).

    Raises
    ------
    AssertionError
        If any m_i < 3, if any m_i is non-integer, or if `kernel` is not
        square with size equal to len(m).

    Notes
    -----
    In block mode the matrix is organized in N+2 block rows/columns:
    block rows 0..N-1 contain U_i and R_i, row N contains P_j, and the last
    block row places `kernel` beneath the P row.
    """
    assert np.all(m > 2), 'The delay lengths `m` must be at least 3.'
    assert np.all(np.mod(m, 1) == 0), 'The delay lengths `m` must be integer.'
    m = m.astype(int)

    # N is the number of delay lines.
    N = len(m)
    assert kernel.shape == (N, N), 'The matrix `A` must be square and have the same size as `m`.'

    if element_wise_assembly:
        # When the number of delay lines is huge, using block_array(AA_blocks) runs into a MemoryError.
        # This approach sets the nonzero elements by hand, without using blocks.
        # Less intuitive, but it works. It is also generally much faster than the alternative.
        M = np.sum(m)
        AA = lil_array((M, M))

        # Figure out which (column) indices are "missing" from the off-diagonal.
        # The way this is used will be clear later.
        skipped_cols = np.cumsum(m-2)

        # This will keep track of how many lines (missing diagonal entries) have been handled.
        handled = 0
        # Iterate over columns of the SSM.
        for i in range(M - (2*N) + 1):
            if i == 0 or i in skipped_cols:
                # This column has NO nonzero element on the off-diagonal (previous row).
                # Instead, the nonzero element on this column is:
                if handled != N:
                    AA[M - (2*N) + handled, i] = 1
                # And the nonzero element on the previous row is:
                if i != 0:
                    AA[i-1, M - N + handled - 1] = 1
                # Increment the number of "skips" that have been handled.
                handled += 1
            else:
                # This column has a nonzero element on the off-diagonal (previous row).
                AA[i-1, i] = 1

        # Finally, insert the (nonzero) elements of the feedback matrix A in their slot.
        # https://stackoverflow.com/a/4319087
        coo_A = coo_array(kernel)
        for i, j, v in zip(coo_A.row, coo_A.col, coo_A.data):
            AA[M - N + i, M - (2*N) + j] = v

        return csr_array(AA)
    else:
        # M is the size of the state transition matrix, in terms of number of blocks.
        # There are N+2 "block rows" and "block columns".
        M = N + 2

        # These are the blocks which make up the state transition matrix.
        # There is one of these for each delay line.
        U = list()
        R = list()
        P = list()
        for i, m_i in enumerate(m):
            # The size of blocks is equal to the delay line length -2.
            block_edge = m_i - 2

            # Construct the U block (inner sample shift).
            U_i = dia_array((np.ones(block_edge), 1), (block_edge, block_edge))
            U.append(csr_array(U_i))

            # Construct the R block (first sample shift).
            R_i = csr_array((block_edge, N))
            R_i[-1, i] = 1
            R.append(R_i)

            # Construct the P block (last sample shift).
            P_i = csr_array((N, block_edge))
            P_i[i, 0] = 1
            P.append(P_i)

        # Prepare a list of lists containing the blocks of AA.
        AA_blocks = list()
        for i in range(M):
            # This list will contain the blocks on the (i)th "block row".
            blocks_row = list()

            for j in range(M):
                # By default, the (i,j)th block is empty.
                block = None

                if i < N:
                    # If the block row index (i) is < N,
                    #  append blocks related to the "innermost" samples of line (i).
                    if j == i:
                        # The blocks on the diagonal are U, which govern shifts from
                        #  the second sample to the second-last sample of the line.
                        block = U[i]
                    elif j == N + 1:
                        # The last block on the row is R, which governs the shift from
                        #  the first sample to the second sample of the line.
                        block = R[i]
                elif i == N and j < N:
                    # If the block row index (i) is = N (second-last block row),
                    #  append blocks related to the last sample of each line.
                    block = P[j]
                elif i == N + 1 and j == N:
                    # If the block row index (i) is = N+1 (last block row), append the feedback matrix,
                    #  governing the dependency of the first sample of each line to the last.
                    block = kernel

                blocks_row.append(block)

            AA_blocks.append(blocks_row)

        # Create a sparse matrix from the blocks.
        return block_array(AA_blocks, format='csr')


def real_positive_search(ssm: csr_array,
                         T60_thresh: float,
                         num_thresh: int,
                         sample_rate: float = 5e3,
                         imaginary_part_thresh: float = 1e-7,
                         raise_on_failure: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find real, positive eigenpairs of a sparse state matrix. Left and right
    eigenvectors are found separately, linked in pairs, and calibrated.

    Performs a shift-invert Arnoldi search (sigma=1) for right eigenpairs of
    the state transition matrix, filters to real positive eigenvalues (by
    bounding the imaginary part to `imaginary_part_thresh`), and stops when:
      1) at least one valid eigenvalue has T60 <= `T60_thresh`, or
      2) at least `num_thresh` valid eigenvalues have been found, or
      3) the search size reaches dim-2 (max allowed by Arnoldi), or
      4) convergence fails.

    The search is then repeated on the transposed matrix to obtain left eigenpairs,
    which are matched to their right counterparts by nearest eigenvalue, discarding
    unmatched pairs. Finally, each left/right pair is calibrated to have unity dot
    product, as discussed in `ART_theory.md`.

    Parameters
    ----------
    ssm : scipy.sparse.csr_array
        Square sparse state transition matrix.
    T60_thresh : float
        Target magnitude threshold for early stopping. Search stops once any
        valid eigenvalue has T60 <= this value.
        Set to None to deactivate this stopping criterion.
    num_thresh : int
        Target count threshold for early stopping. Search stops once at least
        this many valid eigenvalues are found.
        Set to None to deactivate this stopping criterion.
    imaginary_part_thresh : float, default 1e-7
        Maximum absolute imaginary part for an eigenvalue to be treated as real.
    raise_on_failure : bool, default False
        Raise a RuntimeError if no eigenvalues are found.

    Returns
    -------
    numpy.ndarray
        Real eigenvalues (shape (K,)). Values are the average of matched left
        and right eigenvalues.
    numpy.ndarray
        Right eigenvectors (shape (M, K)), columns correspond to eigenvalues.
    numpy.ndarray
        Left eigenvectors (shape (M, K)), columns correspond to eigenvalues.

    Notes
    -----
    - The search size `k` starts at `num_thresh` and doubles up to dim-2.
    - Validity requires positive real part and small imaginary part (as set
      by `imaginary_part_thresh`).
    - Returned count K equals the number of matched valid eigenpairs; it may
      be less than the number initially found on either side.
    """
    if T60_thresh is None and num_thresh is None:
        raise ValueError('The arguments T60_thresh and num_thresh cannot both be None.')

    if T60_thresh is None:
        mag_thresh = None
    else:
        mag_thresh = T60_to_eig(T60_thresh, sample_rate)

    # Start with a right eigenvector search, stopping when
    #   a. we find a (real, positive) pole with T60 < T60_thresh, or
    #   b. we find more than num_thresh (real, positive) poles, or
    #   c. we try looking for more poles than scipy.sparse.eigs() can look for, or
    #   d. the search fails to converge.
    print('\tEigenvalue search (right eigenvectors).')
    if num_thresh is not None:
        k = num_thresh
    else:
        k = 4

    # https://stackoverflow.com/a/46902086
    convergence_failed = False
    while True:
        print('\t\tSearching with', k, 'estimates.')
        try:
            # Perform search in shift-invert mode, centered around sigma=1.
            # This prioritizes eigenvalues closest to 1.
            right_vals, right_vecs = eigs(ssm, k=k, sigma=1.)
        except ArpackNoConvergence as err:
            print('\t\tArpackNoConvergence error')
            print('\t\t\t', err)
            right_vals, right_vecs = err.eigenvalues, err.eigenvectors
            convergence_failed = True

        # Consider only the real, positive eigenvalues.
        valid_idxs = (np.real(right_vals) > 0) & (np.abs(np.imag(right_vals)) < imaginary_part_thresh)
        # TODO: Detect and reject PAIRS of complex eigenvalues with very small imaginary part,
        #       by checking if they are approx. conjugates AND their eigenvectors are approx. conjugates
        # assert np.all(np.isreal(right_vals[valid_idxs]))
        # assert np.all(np.isreal(right_vecs[:, valid_idxs]))
        right_vals = np.real(right_vals[valid_idxs])
        right_vecs = np.real(right_vecs[:, valid_idxs])

        # Consider the "number of located modes" stopping condition.
        if num_thresh is not None:
            print('\t\t\tNumber found / sought: ', len(right_vals), '/', num_thresh)
            if len(right_vals) >= num_thresh:
                break
        # Consider the "lowest located mode" stopping condition.
        if T60_thresh is not None:
            print('\t\t\tLowest found T60 is {:.0f}% of stopping value.'.format(100. * np.log10(mag_thresh) / np.log10(np.min(right_vals))))
            if np.min(right_vals) <= mag_thresh:
                break
        # If the search failed, it has no chance of succeeding at the next loop.
        if convergence_failed:
            break

        # Try again with twice as many candidates. Note that the search locates
        #  values in a radius around 1, including complex values -- increasing
        #  k linearly would make it very slow.
        k *= 2
        # `eigs` has a limitation on the number of requested eigenvalues.
        if k >= ssm.shape[0] - 2:
            break

    # Having found the right eigenpairs, look for the left counterparts.
    # There are some algorithms to find both at the same time, but they
    # are nowhere near as efficient as this approach for huge sparse matrices.
    print('\tEigenvalue search (left eigenvectors).')
    # https://stackoverflow.com/a/46902086
    try:
        # Again, search in shift-invert mode, centered around sigma=1.
        left_vals, left_vecs = eigs(ssm.T, k=k, sigma=1.)
    except ArpackNoConvergence as err:
        print('\t\tArpackNoConvergence error')
        print('\t\t\t', err)
        left_vals, left_vecs = err.eigenvalues, err.eigenvectors

    # Consider only the real, positive eigenvalues.
    valid_idxs = (np.real(left_vals) > 0) & (np.abs(np.imag(left_vals)) < imaginary_part_thresh)
    # TODO: Detect and reject PAIRS of complex eigenvalues with very small imaginary part,
    #       by checking if they are approx. conjugates AND their eigenvectors are approx. conjugates
    # assert np.all(np.isreal(left_vals[valid_idxs]))
    # assert np.all(np.isreal(left_vecs[:, valid_idxs]))
    left_vals = np.real(left_vals[valid_idxs])
    left_vecs = np.real(left_vecs[:, valid_idxs])

    # Match left and right vectors based on their eigenvalues (order is not guaranteed), and rearrange as necessary.
    num_right = len(right_vals)
    num_left = len(left_vals)
    max_num_valid = min(num_right, num_left)
    right_rearrangement = -np.ones(max_num_valid, dtype=int)
    left_rearrangement = -np.ones(max_num_valid, dtype=int)

    num_valid_matches = 0
    if num_left <= num_right:
        # Assign each right value to the corresponding left one.
        for old_left_idx in range(num_left):
            old_right_idx = np.argmin(np.abs(right_vals - left_vals[old_left_idx]))

            if not np.isclose(right_vals[old_right_idx], left_vals[old_left_idx]):
                # No match, invalid pole.
                continue

            if old_right_idx in right_rearrangement:
                warnings.warn('Two right values want to be mapped to the same left value. '
                              + 'Right ' + str(right_vals[old_right_idx]) + ', left ' + str(left_vals[old_left_idx]))

            right_rearrangement[num_valid_matches] = old_right_idx
            left_rearrangement[num_valid_matches] = old_left_idx

            num_valid_matches += 1
    else:
        # Assign each left value to the corresponding right one.
        for old_right_idx in range(num_right):
            old_left_idx = np.argmin(np.abs(left_vals - right_vals[old_right_idx]))

            if not np.isclose(right_vals[old_right_idx], left_vals[old_left_idx]):
                # No match, invalid pole.
                continue

            if old_left_idx in left_rearrangement:
                warnings.warn('Two left values want to be mapped to the same right value. '
                              + 'Right ' + str(right_vals[old_right_idx]) + ', left ' + str(left_vals[old_left_idx]))

            right_rearrangement[num_valid_matches] = old_right_idx
            left_rearrangement[num_valid_matches] = old_left_idx

            num_valid_matches += 1

    if num_valid_matches == 0:
        if raise_on_failure:
            raise RuntimeError('No eigenvalues were shared between left and right.')
        else:
            warnings.warn('No eigenvalues were shared between left and right.')
    if num_valid_matches < max_num_valid:
        warnings.warn('Some eigenvalues were not shared between left and right: '
                      + str(num_valid_matches) + '/' + str(max_num_valid))

    # Remove unassigned (invalid) entries on both sides.
    right_rearrangement = right_rearrangement[:num_valid_matches]
    left_rearrangement = left_rearrangement[:num_valid_matches]

    # Cross-match values and vectors.
    right_vals = right_vals[right_rearrangement]
    right_vecs = right_vecs[:, right_rearrangement]
    left_vals = left_vals[left_rearrangement]
    left_vecs = left_vecs[:, left_rearrangement]

    if not np.allclose(right_vals, left_vals):
        warnings.warn('Even after cross-matching, the left and right eigenvalues mismatch.')
    # Take the average, to slightly improve the numerical accuracy.
    mean_vals = (right_vals + left_vals) / 2

    # Calibrate the full eigenvectors.
    # First, consider a calibration such that their dot products is 1.
    # This makes it so the pair (V, W) forms a valid diagonalization.
    # See "ART_theory.md" for details, specifically the end of section
    #   "MoD-ART eigenvectors".
    calibration = np.einsum('ji,ji->i', left_vecs, right_vecs)
    # Now, we need to add a second calibration, because the eigenvectors are
    #   currently dependent on the sample rate used for decomposition.
    # We want the eigenvector values to be sample-rate-agnostic.
    calibration /= sample_rate
    # Finally, we need to apply the compensation we evaluated.
    # In theory, we could do either:
    #   right_vecs /= calibration
    # or:
    #   left_vecs /= calibration
    # Instead, we split the calibration across both sides, to avoid increasing
    #   or decreasing one side too much (which could cause rounding errors).
    right_vecs /= np.sign(calibration) * np.sqrt(np.abs(calibration))
    left_vecs /= np.sqrt(np.abs(calibration))

    return mean_vals, right_vecs, left_vecs
