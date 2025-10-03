# RAVES preprocessing

### Quick workflow

- Prepare the environment description (input files `materials.csv`, `mesh.mtl`, `mesh.obj`) and 
  place all three files into the same folder. Respect the file format specifications outlined in 
  the following section. The folder's name will be the name of your environment.
- From the `RAVES_Preprocessing` root folder, run `python -m raves 
  "C:/your/environment/folder/path"`. For an example, try `python -m raves 
  ".\example environments\AudioForGames_20_patches"`.
- Copy the files `materials.csv`, `mesh.mtl`, `mesh.obj`, `MoD-ART.csv`, and `path_indexing.mtx` 
  into your Unity project assets, matching the folder structure shown below. Follow the 
  instructions of the `RAVES-Unity` repository to set up RAVES in the Unity project.

Alternatively, you may run the individual scripts `compute_ART`, `compute_MoDART`, and 
`update_ART_materials`, or import them as you would a package, e.g. `from raves import compute_ART`.

### Unity project folder structure

Place your mesh files in `PythonExports/YourEnvironmentName/` as shown in the following example.
A file named `ProcessedPrefabs/YourEnvironmentName.prefab` will be automatically generated the first
time your environment is loaded in the Unity editor. Make sure the `RAVES-Unity` GitHub
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
not to change any other part of `mesh.mtl` and `mesh.obj`: **the mesh geometry and 
patch-triangle grouping must remain unchanged w.r.t. the original ART computation.**
After making your changes, run the `update_ART_materials` script, providing the path to your 
environment folder as input.

### Design considerations

The effects of surface patch design on ART simulation results are still understudied, but we
recommend a few rules-of-thumb:
- each surface patch should lie on a single plane and form a polygon with no holes;
- each surface patch should be relatively isotropic (avoid long and narrow polygons);
- all surface patches should have roughly the same area.

The `compute_ART` script determines the surface integral sampling (points-per-meter-squared) 
based on the area of the smallest surface patch. If the computation takes a long time, try 
using larger patches, or increase the minimum number of sample points per patch. The former 
approach will decrease the directional resolution of the ART model, while the latter will 
decrease the numerical accuracy of the surface integrals.

## Input files

### mesh.obj

The mesh geometry should be provided in basic Wavefront format (`mesh.obj`+`mesh.mtl`).
The first line of `mesh.obj` should be `mtllib mesh.mtl`, with `mesh.mtl` being placed in the same
folder as `mesh.obj`. All following lines should be restricted to vertex definitions `v`, face 
definitions `f`, comments `#`, and materials assignments `usemtl`, as detailed in the following.
Any other lines will be ignored by the acoustic analysis, but may alter the visual appearance of the
mesh in Unity.

Vertex definition lines start with `v` followed by three floating-point values, separated by spaces.
These are the vertex's coordinates in three-dimensional space. Optionally, the line may include 
an inline comment (starting with `#`) specifying the vertex index. These comments only serve as 
a human-readable reference, and are ignored by the parser. Avoid using this type of "inline" 
comment on lines starting with `f` or `usemtl`.

Face definition lines start with `f` followed by three integer numbers, separated by spaces.
These are the indices of the vertices forming each face, listed counter-clockwise around the 
surface normal. While the Wavefront format supports polygonal faces with more than three 
vertices, **our code only supports triangles**. The vertex indices refer to the order in which 
vertices are defined in the same file; note that the indexing starts from 1.

Comment lines start with `#` and are ignored by all parsers. You may use these to label groups 
of surface patches, e.g., `# Room 1 floor`. Blank lines are also ignored.

Material assignment lines must follow the format `usemtl Patch_{i}_Mat_{material}`, where `i` is the
ART surface patch index and `material` is a string identifying the surface material.
**The patch indices should range from 1 to the number of patches. The material identifier must 
only contain ASCII letters, digits, or underscores.**
Each `usemtl` line applies to all faces defined in following lines, until the next `usemtl` 
line. In the example below, each surface patch is a rectangle formed by two adjacent triangles.

#### Example

```
mtllib mesh.mtl

################################ Vertices

v 0.0 0.0 0.0                  # Vertex 1
v 0.0 3.0 0.0                  # Vertex 2
[...]
v -10.0 3.0 13.0               # Vertex 28

################################ Faces

################################ Room 1 floor
usemtl Patch_1_Mat_Carpet
f 15 3 1
f 5 15 1
################################ Room 1 walls
usemtl Patch_2_Mat_Bricks_open_joints
f 4 16 6
f 2 4 6
[...]
################################ Room 3 walls
usemtl Patch_20_Mat_Concrete_painted
f 28 20 19
f 27 28 19
```

### mesh.mtl

The `mesh.mtl` file should contain a definition for each patch ID string `Patch_{i}_Mat_{material}`
mentioned in `mesh.obj`. Each definition consists of two lines: the first one is `newmtl Patch_
{i}_Mat_{material}`, and the second one is `Kd <red> <green> <blue>` specifying the RGB color of 
the patch. **Note that these strings must match exactly the ones given in `mesh.obj`.**

The material properties defined in `mesh.mtl` are purely visual, and have no bearing on the acoustic
processing. This file only serves to ensure that mesh materials are imported correctly in Unity.
Unity disregards the materials mentioned in the `.obj` if they have no matching definition in
the `.mtl`. Nevertheless, these are the colors that will be displayed in the Unity editor, so 
you may find them useful for visual validation of the patch assignment.

#### Example

```
newmtl Patch_1_Mat_Carpet
Kd 1.0 0.7849 0.302
newmtl Patch_2_Mat_Bricks_open_joints
Kd 0.8084 0.7571 0.0644
[...]
newmtl Patch_20_Mat_Concrete_painted
Kd 0.098 0.6756 1.0
```

### materials.csv

The first line of `materials.csv` should report the center frequencies of the desired octave bands.
These must form a contiguous range of valid octave bands.

Following lines report each material's absorption and scattering coefficients.
There must be exactly two lines for each material; the first holds the absorption coefficients, and
the second holds the scattering coefficients. Each coefficient line, after the material 
identifier, may either present a single value or as many values as there are frequency bands.
In the former case, the same value is applied in all frequency bands. In the example below, 
absorption coefficients are specified for each band, whereas scattering coefficients are 
frequency-independent.

#### Example

```
Frequencies, 125.0 250.0 500.0 1000.0 2000.0 4000.0 8000.0 16000.0
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

TODO: Explanation will go here.
TODO: Make sure to specify the definition (power, not radiance).

- `ART_kernel_diffuse.mtx`
    - TODO: Explanation will go here.
- `ART_kernel_specular.mtx`
    - TODO: Explanation will go here.
- `ART_octave_band_1.mtx`, `ART_octave_band_2.mtx`, ...
    - TODO: Explanation will go here.

### Propagation paths

TODO: Explanation will go here.
TODO: Double-check the accuracy of `<start patch idx> <end patch idx> <propagation path idx>`.

- `path_indexing.mtx`
    - TODO: Sparse, square, integer-valued matrix...
- `path_lengths.csv`
    - TODO: Propagation path lengths, in meters.
    - TODO: Only reports paths with visibility.

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

TODO: Explanation will go here.
TODO: Make sure to specify the definition (power, not radiance).
TODO: Make sure to specify the definition (recursive loop I/O layout).
TODO: Make sure to specify that several terms are already baked into the eigenvectors, making 
their combination with RTM results easier.

#### Example

```
1, 1.6473734012319403
0.009305340148100468, 0.009166827596212134, [...], 0.005161508412455599
0.06801976899109753, 0.0764863885656201, [...], 0.005676870715431933
[...]
8, 0.1562517087118225
-0.012946122899032202, -0.011744174832778655, [...], -0.008311433932984587
-0.10343713733055754, -0.10050090598321827, [...], -0.00039810214464333296
```
