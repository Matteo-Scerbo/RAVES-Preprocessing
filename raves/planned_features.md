# TODO

- Improve and validate the surface integral.
  Use a solid angle integral from each surface sample point.
- Revise all comments and documentation
- Add `run_ART.py`

### `io.py`
- All necessary parsers and writers

### `compute_ART.py`
- Load `mesh.obj` into a `TriangleMesh`; store the material name of each patch for later use
- Initialize `path_lengths`, `kernel_diffuse`, and `kernel_specular` (sparse arrays)
- Perform surface integrals. For each patch:
  - Get patch areas as sum of triangle areas. Uniformly sample the patch surface. For each sample 
    point, trace rays in visible hemisphere; accumulate and normalize as in existing code. Note 
    that etendue can be inferred from patch areas and `kernel_diffuse`.
- Write `ART_kernel_diffuse.mtx`, `ART_kernel_specular.mtx`, `path_lengths.csv`
- Read `materials.csv`, prepare and write `ART_octave_band_1.mtx`, `ART_octave_band_2.mtx`, etc.

### `update_ART_materials.py`
- Read `mesh.obj` and store the material name of each patch; read `materials.csv`
- Read `ART_kernel_diffuse.mtx`, `ART_kernel_specular.mtx`
- Prepare and write `ART_octave_band_1.mtx`, `ART_octave_band_2.mtx`, etc.

### `compute_MoDART.py`
- Read `ART_kernel_diffuse.mtx`, `ART_kernel_specular.mtx`, `path_lengths.csv`
- Perform real-valued decomposition as in existing code
- Rescale eigenvectors; note that etendue can be inferred from patch areas and `kernel_diffuse`
- Write `MoD-ART.csv`
