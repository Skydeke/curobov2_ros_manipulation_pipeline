# Jetson zero-copy voxel tensor via POSIX shared memory + cudaHostRegister.
#
# Producer:
#   1. shm_open(shm_name, O_CREAT|O_RDWR) + ftruncate(nbytes) + mmap
#   2. cudaHostRegister(ptr, nbytes, Mapped|Portable)
#   3. cudaHostGetDevicePointer(...) -> dev_ptr; wrap as torch tensor on cuda
#   4. Write the tensor on GPU; cudaDeviceSynchronize before publishing
#
# Consumer:
#   On first message OR when buffer_epoch / producer_pid change:
#     shm_open(shm_name, O_RDWR) + mmap(nbytes)
#     cudaHostRegister(ptr, nbytes, Mapped|Portable)
#     Read directly from the mmap (host side) — Tegra unifies host/device RAM,
#     so reads see the producer's GPU writes without an explicit copy.
#
# Cross-container requirements (same Jetson host):
#   Either same container, or bind-mount /dev/shm and use --ipc=host.
#
# Convention:
#   1.0 (float16) = free voxel
#   0.0 (float16) = occupied voxel
#   (lets the consumer feed the U-Net directly with no inversion step)

std_msgs/Header header

# Bbox metadata in robot base frame (matches SparseVoxelGrid for consistency).
geometry_msgs/Vector3 origin
float32 resolution
uint32 size_x
uint32 size_y
uint32 size_z

# Tensor layout.
string dtype          # e.g. "float16"
uint32[] shape        # e.g. [1, 1, 128, 128, 128]
uint64 nbytes         # total byte count of the shared buffer

# POSIX shm name, e.g. "/cuda_voxel_<pid>". Open with shm_open() on consumer.
string shm_name

# Incremented by the producer whenever the shared buffer is (re)allocated
# (e.g. bbox resized). Consumer must close/reopen when this value changes.
uint64 buffer_epoch

# Monotonic frame counter. Consumer can detect dropped frames.
uint64 frame_id

# Producer process id. If it changes, the buffer was reallocated too — treat
# as a new epoch and reopen.
int32 producer_pid
