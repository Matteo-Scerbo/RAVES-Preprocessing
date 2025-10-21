import warnings
import numpy as np
from typing import Tuple

from scipy.sparse import coo_array, csr_array, lil_array, dia_array, block_array
from scipy.sparse.linalg import ArpackNoConvergence, eigs


def build_ssm(kernel: csr_array, m: np.ndarray,
              element_wise_assembly: bool = True
              ) -> csr_array:
    # TODO: Fill out documentation properly.
    """
    Construct the state transition matrix of a system exhibiting a "Feedback Delay Network" structure.
    Args:
        A:
        m:
        element_wise_assembly:

    Returns:

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
        return block_array(AA_blocks, format='csc')


def real_positive_search(ssm: csr_array,
                         mag_thresh: float,
                         num_thresh: int,
                         imaginary_part_thresh: float = 1e-7
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Fill out documentation properly.
    """

    Args:
        ssm:
        mag_thresh:
        num_thresh:
        imaginary_part_thresh:

    Returns:

    """
    # Start with a right eigenvector search, stopping when
    #   a. we find a (real, positive) pole with T60 < t60_thresh, or
    #   b. we find more than num_thresh (real, positive) poles, or
    #   c. we try looking for more poles than scipy.sparse.eigs() can look for, or
    #   d. the search fails to converge.
    print('\tEigenvalue search (right eigenvectors).')
    # https://stackoverflow.com/a/46902086
    k = num_thresh
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

        print('\t\t\tLowest found / sought: ', np.min(right_vals), '/', mag_thresh, '(T60 ratio ', np.log10(np.min(right_vals)) / np.log10(mag_thresh), ')')
        print('\t\t\tNumber found / sought: ', len(right_vals), '/', num_thresh)

        if np.min(right_vals) <= mag_thresh:
            break
        if len(right_vals) >= num_thresh:
            break
        if k == ssm.shape[0] - 2:
            break
        if convergence_failed:
            break

        k *= 2
        k = min(k, ssm.shape[0] - 2)

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

    # Calibrate the full eigenvectors: their dot products should be 1.
    calibration = np.einsum('ji,ji->i', left_vecs, right_vecs)
    # In theory, we could do either:
    #   right_vecs /= calibration[np.newaxis]
    # or:
    #   left_vecs /= calibration[np.newaxis]
    # Instead, split the calibration across both sides, to avoid increasing/decreasing one side too much.
    right_vecs /= np.sign(calibration) * np.sqrt(np.abs(calibration))
    left_vecs /= np.sqrt(np.abs(calibration))

    return mean_vals, right_vecs, left_vecs
