#include "../utils.h"
#include "operators.h"
#include "pooling.h"

#ifdef ENABLE_CPU
#include "cpu/pooling_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/pooling.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif
#ifdef ENABLE_MT_GPU
#include "musa/pooling_musa.h"
#endif

__C infiniopStatus_t infiniopCreatePoolingDescriptor(
    infiniopHandle_t handle,
    infiniopPoolingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    uint64_t const *kernel_shape,
    uint64_t const *pads,
    int64_t const *strides,
    uint64_t n,
    int pooling_type) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreatePoolingDescriptor(handle, (PoolingCpuDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreatePoolingDescriptor((CudaHandle_t) handle, (PoolingCudaDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaCreatePoolingDescriptor((MusaHandle_t) handle, (PoolingMusaDescriptor_t *) desc_ptr, y, x, kernel_shape, pads, strides, n, pooling_type);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetPoolingWorkspaceSize(infiniopPoolingDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetPoolingWorkspaceSize((PoolingCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetPoolingWorkspaceSize((PoolingCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO

#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaGetPoolingWorkspaceSize((PoolingMusaDescriptor_t) desc, size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopPooling(infiniopPoolingDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuPooling((PoolingCpuDescriptor_t) desc, workspace, workspace_size, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaPooling((PoolingCudaDescriptor_t) desc, workspace, workspace_size, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaPooling((PoolingMusaDescriptor_t) desc, workspace, workspace_size, y, x, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyPoolingDescriptor(infiniopPoolingDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyPoolingDescriptor((PoolingCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyPoolingDescriptor((PoolingCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaDestroyPoolingDescriptor((PoolingMusaDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
