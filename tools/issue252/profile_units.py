"""PyTorch profiler duration normalization (its native duration unit is us)."""

def operator_record(event):
    device = getattr(event,'self_device_time_total',None)
    if device is None:
        device = getattr(event,'self_cuda_time_total',0.)
    return {'name':event.key,'calls':event.count,
            'self_cpu_seconds':float(event.self_cpu_time_total)/1e6,
            'self_device_seconds':float(device)/1e6}
