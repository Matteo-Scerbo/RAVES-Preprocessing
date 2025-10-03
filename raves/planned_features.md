### Planned features

#### What goes where?
`raytracing` module:
- TriangleMeshSoA class
- RayBundleSoA class (methods for tracing; works as pencil or single ray by using broadcasting)

`io` module:
- All of the parsers and writers

#### Avoid using classes where it's not required.
We probably don't need a model class.
We probably don't need node classes.
We definitely don't need the propagation line class.

#### Polygon format
Replace the polygon class with a TriangleMeshSoA, translating back from C++.

#### RTM algorithm
Replace the ray-tracing functions with triangle-based algorithms, translated back from C++.

#### Validate, improve, and replace the surface integral
Use a solid angle integral from each surface sample point.
