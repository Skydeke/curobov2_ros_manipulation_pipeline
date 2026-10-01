# Sparse voxel grid: only the linear indices of occupied cells are sent.
# Dense reconstruction:
#   dense[x, y, z] = 1  where  linear = x * size_y * size_z + y * size_z + z
# All other cells are zero (free).

std_msgs/Header header

# Bbox origin (metres, robot base frame).
geometry_msgs/Vector3 origin

# Cubic voxel side length (metres).
float32 resolution

# Grid dimensions in voxels.
uint32 size_x
uint32 size_y
uint32 size_z

# Flattened C-order indices of occupied voxels.
int32[] occupied_indices
