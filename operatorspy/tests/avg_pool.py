from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch
from typing import Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class AvgPoolDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopAvgPoolDescriptor_t = POINTER(AvgPoolDescriptor)


def pool(x, k, padding, stride, dilation = 1):
    pooling_layers = {
        1: torch.nn.AvgPool1d,
        2: torch.nn.AvgPool2d,
        3: torch.nn.AvgPool3d,
    }

    ndim = len(x.shape) - 2
    if ndim not in pooling_layers:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None

    if ndim == 3 and x.dtype == torch.float16:
        ans = pooling_layers[ndim](k, stride=stride, padding=padding)(x.to(torch.float32)).to(torch.float16)
    else:
        ans = pooling_layers[ndim](k, stride=stride, padding=padding)(x)
    if PROFILE:
        torch.cuda.synchronize()
    return ans


def inferShape(x_shape, kernel_shape, padding, strides):
    assert (
        len(x_shape) - 2 == len(kernel_shape) == len(padding) == len(strides)
    ), "kernel, pads, and strides should have the same length; the length of input x should be 2 more than that of kernel"
    input_shape = x_shape[2:]
    output_shape = []

    for dim, k, p, s in zip(input_shape, kernel_shape, padding, strides):
        output_dim = (dim + 2 * p - k) // s + 1
        output_shape.append(output_dim)

    return x_shape[:2] + tuple(output_shape)

# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape, 
    k_shape, 
    padding,
    strides,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing AvgPool on {torch_device} with x_shape:{x_shape} kernel_shape:{k_shape} padding:{padding} strides:{strides} dtype:{tensor_dtype}"
    )

    x = torch.ones(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.zeros(inferShape(x_shape, k_shape, padding, strides), dtype=tensor_dtype).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = pool(x, k_shape, padding, strides)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = pool(x, k_shape, padding, strides)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopAvgPoolDescriptor_t()

    check_error(
        lib.infiniopCreateAvgPoolDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_void_p(k_shape),
            tuple_to_void_p(padding),
            tuple_to_void_p(strides),
            len(k_shape),
        )
    )

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetAvgPoolWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopAvgPool(
                descriptor,
                workspace_ptr,
                workspaceSize,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopAvgPool(
                    descriptor,
                    workspace_ptr,
                    workspaceSize,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")


    print(x)
    print(y)
    print(ans)
    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyAvgPoolDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides in test_cases:
        test(lib, handle, "cpu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides in test_cases:
        test(lib, handle, "cuda", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides in test_cases:
        test(lib, handle, "mlu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

def test_musa(lib, test_cases):
    import torch_musa

    device = DeviceEnum.DEVICE_MUSA
    handle = create_handle(lib, device)
    for x_shape, kernel_shape, padding, strides in test_cases:
        # test(lib, handle, "musa", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float16)
        test(lib, handle, "musa", x_shape, kernel_shape, padding, strides, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, kernel_shape, padding, strides
        # ((1, 1, 10), (3,), (1,), (1,)),
        ((1, 1, 2, 2), (2, 2), (1, 1), (1, 1)),
        ((32, 4, 224, 224), (3, 3), (1, 1), (2, 2)),
        ((1, 1, 16, 16, 16), (5, 5, 5), (2, 2, 2), (2, 2, 2)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateAvgPoolDescriptor.restype = c_int32
    lib.infiniopCreateAvgPoolDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAvgPoolDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
    ]
    lib.infiniopGetAvgPoolWorkspaceSize.restype = c_int32
    lib.infiniopGetAvgPoolWorkspaceSize.argtypes = [
        infiniopAvgPoolDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopAvgPool.restype = c_int32
    lib.infiniopAvgPool.argtypes = [
        infiniopAvgPoolDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyAvgPoolDescriptor.restype = c_int32
    lib.infiniopDestroyAvgPoolDescriptor.argtypes = [
        infiniopAvgPoolDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.musa:
        test_musa(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.musa):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
