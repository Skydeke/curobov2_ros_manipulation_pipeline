std_msgs/Header header

# Monotonically increasing counter, reset at node start.
uint32 frame_id

# Per-stage wall-clock durations in milliseconds.
# Scatter:       sparse index → dense GPU tensor
# Resize:        invert occupancy, cast, unsqueeze
# GPU sync:      torch.cuda.synchronize() after data prep
# Prediction:    TRT / PyTorch UNet3D forward pass
# CPU transfer:  move result tensor to CPU
float64 t_scatter_ms
float64 t_resize_ms
float64 t_gpu_sync_ms
float64 t_prediction_ms
float64 t_cpu_transfer_ms
float64 t_total_ms

# Instantaneous throughput derived from t_total_ms.
float64 fps
