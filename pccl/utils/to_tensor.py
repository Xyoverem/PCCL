import torch
import ctypes

def to_tensor(s: str):
    bytes_str = s.encode('utf-8')
    return torch.frombuffer(bytes_str, dtype=torch.uint8)

def to_str(t: torch.Tensor):
    cpu_t = t.cpu()
    ptr = cpu_t.data_ptr()
    size = t.numel() * t.element_size()
    buffer = (ctypes.c_ubyte * size).from_address(ptr)
    s_new = bytes(buffer).decode('utf-8')
    return s_new
