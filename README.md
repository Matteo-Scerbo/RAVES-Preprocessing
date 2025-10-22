# RAVES preprocessing

## Usage

You can either import the main functions into your own Python script, or run the preprocessing 
from the command line. To import in a Python script, clone the repository somewhere that your 
Python script can see, and use one or more of the following lines:
```
from raves import raves
from raves import compute_ART
from raves import compute_MoDART

raves("path/to/your/environment/folder")
```
To launch from the command line, run
```
python -m raves "path/to/your/environment/folder"
```
from the root directory of the repository.

### Quick workflow

- Prepare the environment description (input files `materials.csv`, `mesh.mtl`, `mesh.obj`) 
  following the file format outlined in the following section. Place all three files into the 
  same folder (the folder's name will be the name of your environment).
- Run the `raves` script, either from a command line console or from a Python script of your own,
  as shown above. In either case, the first argument should be the path to your environment folder.
- The files `path_indexing.mtx` and `MoD-ART.csv` (among others) will be created in your 
  environment folder. Copy the five files `materials.csv`, `mesh.mtl`, `mesh.obj`,
  `path_indexing.mtx`, and `MoD-ART.csv` into your Unity project assets, matching the folder 
  structure shown below.

### Unity project folder structure

Place your mesh files in `PythonExports/YourEnvironmentName/` as shown in the following example.
The first time your environment is loaded in the Unity editor, a file named
`YourEnvironmentName.prefab` will be automatically generated in `ProcessedPrefabs`.
Make sure the `RAVES-Unity` GitHub
repository is cloned in the `Assets` folder as shown.

```
UnityProjectName/
├── Assets/
│   ├── RAVES-Unity/
│   │   └── [cloned GitHub repository]
│   ├── Resources/
│   │   ├── ProcessedPrefabs/
│   │   │   ├── YourEnvironmentName.prefab
│   │   │   ├── YourOtherEnvironmentName.prefab
│   │   │   └── [...]
│   │   ├── PythonExports/
│   │   │   ├── YourEnvironmentName/
│   │   │   │   ├── materials.csv
│   │   │   │   ├── mesh.mtl
│   │   │   │   ├── mesh.obj
│   │   │   │   ├── MoD-ART.csv
│   │   │   │   └── path_indexing.csv
│   │   │   ├── YourOtherEnvironmentName/
│   │   │   │   ├── materials.csv
│   │   │   │   ├── mesh.mtl
│   │   │   │   ├── mesh.obj
│   │   │   │   ├── MoD-ART.csv
│   │   │   │   └── path_indexing.csv
│   │   │   └── [...]
│   │   └── [...]
│   └── [...]
└── [...]
```

### Modifying the materials of an existing ART model

If you wish to modify the surface material properties after running `compute_ART` but
before running `compute_MoDART`, it is possible to do so without having to re-run the most
expensive ART computations. Simply modify the contents of `materials.csv` and, if necessary, the
material `Mat_{material}` assigned to each surface patch in `mesh.mtl` and `mesh.obj`. Take care
not to change any other part of `mesh.mtl` and `mesh.obj`: **the mesh geometry and surface patches
must remain unchanged w.r.t. the original ART computation, including the patch indices.**
After making your changes, simply run the `compute_ART` script again. It will detect the
existing kernel files (which are independent of material properties) and use them alongside the
new material data, instead of performing the numerical surface integration again.
If you actually want to overwrite the existing numerical integration results, run the script with 
argument `--overwrite`.

### Design considerations

The effects of surface patch design on ART simulations are still understudied, but we recommend 
a few rules-of-thumb:
- each surface patch should form a single polygon (coplanar and connected triangles);
- each surface patch should be relatively compact (avoid long and narrow polygons);
- all surface patches should have roughly the same area.

The quality of the numerical integral in `compute_ART` is controlled by two variables: the 
number of surface sample points per square meter, and the number of rays in a hemisphere (i.e., 
number of direction sample points per $2\pi$ steradian).
The functions `assess_ART` and `assess_ART_on_grid` are provided as tools to assess the 
numerical accuracy of integrals for different values of these parameters.

If the computation takes a long time, try using larger patches, or decrease the density of 
sample points and rays. The former approach will decrease the directional resolution of the ART 
model, while the latter will decrease the numerical accuracy of the surface integrals.

## Input files

### mesh.obj

The mesh geometry should be provided in basic Wavefront format (`mesh.obj`+`mesh.mtl`).
The first line of `mesh.obj` should be `mtllib mesh.mtl`, with `mesh.mtl` being placed in the same
folder as `mesh.obj`. All other lines of `mesh.obj` should be restricted to vertex definitions
`v`, face definitions `f`, comments `#`, and materials assignments `usemtl`, as detailed in the 
following. Any other lines will be ignored by the acoustic analysis, but may alter the visual 
appearance of the mesh in Unity.

Vertex definition lines start with `v` followed by three floating-point values, separated by spaces.
These are the vertex's three-dimensional coordinates. Optionally, the line may include an inline
comment (starting with `#`) specifying the vertex index. Note that vertex indices start from 1.
These comments only serve as a human-readable reference, and are ignored by the parser. Avoid 
using this type of "inline" comment on lines starting with `f` or `usemtl`: they create trouble 
for the Unity parser.

Face definition lines start with `f` followed by three integer numbers, separated by spaces.
These are the indices of the vertices forming each face, listed counter-clockwise around the 
surface normal. While the Wavefront format supports polygonal faces with more than three 
vertices, **our code only supports triangles**. The vertex indices refer to the order in which 
vertices are defined in the same file.

Comment lines start with `#` and are ignored by all parsers. You may use these to label groups 
of surface patches, e.g., `# Room 1 floor`. Blank lines are also ignored.

Material assignment lines must follow the format `usemtl Patch_{i}_Mat_{material}`, where `i` is the
ART surface patch index and `material` is a string identifying the surface material.
**The patch indices should range from 1 to the number of patches. The material identifier must 
only contain ASCII letters, digits, or underscores.**
Each `usemtl` line applies to all faces defined in following lines, until the next `usemtl` 
line. In the example below, each surface patch is a rectangle formed by two adjacent triangles.
**Our code requires all triangles in each patch to be coplanar.**

#### Example

```
mtllib mesh.mtl

################################ Vertices

v 0.0 0.0 0.0                  # Vertex 1
v 0.0 3.0 0.0                  # Vertex 2
[...]
v -10.0 3.0 13.0               # Vertex 28

################################ Faces

# Room 1 floor
usemtl Patch_1_Mat_Carpet
f 15 3 1
f 5 15 1
# Room 1 walls
usemtl Patch_2_Mat_Bricks_open_joints
f 4 16 6
f 2 4 6
[...]
# Room 3 walls
usemtl Patch_20_Mat_Concrete_painted
f 28 20 19
f 27 28 19
```

### mesh.mtl

The `mesh.mtl` file should contain a definition for each patch ID string `Patch_{i}_Mat_{material}`
which appears in `mesh.obj`. **Note that these strings must match exactly the ones given in 
`mesh.obj`.** Each definition consists of two lines: the first one is
`newmtl Patch_{i}_Mat_{material}`, and the second one is `Kd <red> <green> <blue>` specifying 
the (diffuse) RGB color of the patch. More lines may be added to specify other visual properties,
but the two lines above are the bare minimum for correct parsing.

The material properties defined in `mesh.mtl` are purely visual, and have no bearing on the acoustic
processing. This file only serves to ensure that mesh materials are imported correctly in Unity.
Unity disregards the materials mentioned in the `.obj` if they have no matching definition in
the `.mtl`. Nevertheless, these are the visual properties that will be displayed in the Unity 
editor, so you may find them useful for visual validation of the patch assignment.

#### Example

```
newmtl Patch_1_Mat_Carpet
Kd 1.0 0.7232 0.102
newmtl Patch_2_Mat_Bricks_open_joints
Kd 0.9289 0.8723 0.1074
[...]
newmtl Patch_20_Mat_Concrete_painted
Kd 0.0 0.5751 0.898
```

### materials.csv

The first line of `materials.csv` should start with `Frequencies` and report the center frequencies 
of the desired bands. These values must form a contiguous range of valid octave bands.

Following lines report each material's absorption and scattering coefficients.
There must be exactly two lines for each material; the first holds the absorption coefficients, and
the second holds the scattering coefficients. Each coefficient line may either present as many 
floating point values as there are frequency bands, or a single floating point value.
In the latter case, the same value is applied in all frequency bands. In the example below, 
absorption coefficients are specified for each band, whereas scattering coefficients are 
frequency-independent.

#### Example

```
Frequencies, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0
Carpet, 0.05, 0.13, 0.6, 0.24, 0.28, 0.32, 0.32, 0.32
Carpet, 0.3
Bricks_open_joints, 0.07, 0.38, 0.21, 0.15, 0.25, 0.31, 0.31, 0.31
Bricks_open_joints, 0.3
Marble_or_tile, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02
Marble_or_tile, 0.3
Wood_flooring, 0.15, 0.11, 0.1, 0.07, 0.06, 0.07, 0.07, 0.07
Wood_flooring, 0.3
Concrete_painted, 0.1, 0.05, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08
Concrete_painted, 0.3
```

## Output files

### Scattering matrices

Scattering matrices are stored in Matrix Market format (`.mtx`), because they are very sparse.
Note that these matrices are based on the ART formulation where the quantity 
traversing propagation paths is *power*, as opposed to *averaged radiance*. This 
aspect is discussed further in `ART_theory.md`.

The `compute_ART` script writes scattering matrices in the following files:
- `ART_kernel_diffuse.mtx`
    - Diffuse (Lambertian) component of the scattering matrix: no energy losses, all scattering 
      coefficients set to 1.
- `ART_kernel_specular.mtx`
    - Specular component of the scattering matrix: no energy losses, all scattering 
      coefficients set to 0.
- `ART_kernel_<band_index>.mtx` for each frequency band
    - These are the actual scattering matrices, per frequency band. Frequency band incides start 
      from 1. The diffuse and specular components are weighted based on the scattering 
      coefficient of each patch in the specified frequency band, summed together, and then 
      scaled based on the absorption coefficient of each patch in the specified frequency band. 
      Air absorption losses are also included in the scattering matrix, based on propagation 
      path lengths.

Note that the rows and columns of all these matrices follow the propagation path indexing 
specified in `path_indexing.mtx` (see below).

### Propagation paths

The `compute_ART` script writes propagation path details in the following files:
- `path_indexing.mtx`
    - Sparse, square, integer-valued matrix relating each pair of surface patch indices to the
      index of a propagation path. Zero elements (not reported in the file) denote invalid paths;
      patch and path indices both start from 1. This is one of the files which should be copied 
      in the Unity asset folder.
- `path_delays.csv`
    - Propagation path delays, in seconds. Given by the path lengths (see below) divided by the 
      sound speed, which is computed based on the given temperature.
- `path_lengths.csv`
    - Propagation path lengths, in meters. Defined as the average distance between pairs of 
      points in the double surface integral between the two surface patches at either end of the 
      path.
- `path_etendues.csv`
    - Propagation path etendues, i.e., product of projected area and solid angle integrated
      between the two surface patches at either end of the path. It is required to prepare 
      the exported MoD-ART eigenvectors to be used in Unity. This aspect is discussed further in 
      `ART_theory.md`.

Note that `path_lengths.csv` and `path_etendues.csv` only report valid paths, following the 
indices in `path_indexing.mtx`.

#### Example `path_indexing.mtx`

```
%%MatrixMarket matrix coordinate integer general
%Path indexing of ART model: <start patch idx> <end patch idx> <propagation path idx>
20 20 208
1 2 13
1 3 25
[...]
20 19 199
```

### MoD-ART.csv

The `compute_MoDART` script writes energy decay mode details in the `MoD-ART.csv` and
`MoD-ART extra.csv` files. In the former, the number of modes in each frequency band is capped 
at a given threshold, whereas the latter also includes any additional modes found above the limit.
Each mode is characterized by three consecutive lines. The first line of each mode contains only 
two elements: an integer and a floating point value. The integer value is the index of the 
frequency band to which this mode pertains (starting from 1, same as `ART_kernel_<band_index>.mtx`).
The floating point value is the $T_{60}$ of this mode, in seconds. The second and third lines of 
each mode contain the right and left eigenvectors, respectively.

The eigenvectors reported here are not defined exactly as in the papers. In preparation for 
their use at runtime, the right eigenvector is left-multiplied by the scattering matrix, and 
then divided by the path etendues. Both vectors are multiplied by $4 \pi$. The reasons for these 
weights are discussed in `ART_theory.md`.

By default (this can be disabled), the `compute_MoDART` script also creates an image file
`MoD-ART T60 values.png`, which illustrates the $T_{60}$ values of all discovered energy modes 
per frequency band. The $T_{60}$ values are plotted on a log scale, in seconds.

#### Example `MoD-ART.csv`

```
1, 1.6473734012319403
0.009305340148100468, 0.009166827596212134, [...], 0.005161508412455599
0.06801976899109753, 0.0764863885656201, [...], 0.005676870715431933
[...]
8, 0.1562517087118225
-0.012946122899032202, -0.011744174832778655, [...], -0.008311433932984587
-0.10343713733055754, -0.10050090598321827, [...], -0.00039810214464333296
```
