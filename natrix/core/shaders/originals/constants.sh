#ifndef CONSTANTS_SH_HEADER_GUARD
#define CONSTANTS_SH_HEADER_GUARD

// workgroup size of the culling compute shader
// D3D compute shaders only allow up to 1024 threads per workgroup
// GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS also only guarantees 1024
#define GROUP_SIZE 16

#define VELOCITY_IN 1
#define VELOCITY_OUT 2
#define PRESSURE_IN 3
#define PRESSURE_OUT 4
#define VORTICITY 5
#define DIVERGENCE 6
#define OBSTACLES 7
#define GENERIC 8
#define PARTICLES_IN 9
#define PARTICLES_OUT 10

#endif // CONSTANTS_SH_HEADER_GUARD