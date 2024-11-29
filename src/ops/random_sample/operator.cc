#include "../utils.h"
#include "operators.h"
#include "ops/random_sample/random_sample.h"

#ifdef ENABLE_CPU
#include "cpu/random_sample_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/random_sample.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/random_sample_bang.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/random_sample_aclnn.h"
#endif
#ifdef ENABLE_MT_GPU
#include "musa/random_sample_musa.h"
#endif

__C infiniopStatus_t infiniopCreateRandomSampleDescriptor(infiniopHandle_t handle, infiniopRandomSampleDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result, infiniopTensorDescriptor_t probs) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateRandomSampleDescriptor(handle, (RandomSampleCpuDescriptor_t *) desc_ptr, result, probs);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaCreateRandomSampleDescriptor((CudaHandle_t) handle, (RandomSampleCudaDescriptor_t *) desc_ptr, result, probs);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateRandomSampleDescriptor((BangHandle_t) handle,
                                                    (RandomSampleBangDescriptor_t *) desc_ptr, result,
                                                    probs);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateRandomSampleDescriptor((AscendHandle_t) handle,
                                                     (RandomSampleAclnnDescriptor_t *) desc_ptr, result, probs);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu:
            return musaCreateRandomSampleDescriptor((MusaHandle_t) handle, (RandomSampleMusaDescriptor_t *) desc_ptr, result, probs);
#endif
    }
    return STATUS_BAD_DEVICE;
};

__C infiniopStatus_t infiniopGetRandomSampleWorkspaceSize(infiniopRandomSampleDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetRandomSampleWorkspaceSize((RandomSampleCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetRandomSampleWorkspaceSize((RandomSampleCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetRandomSampleWorkspaceSize((RandomSampleBangDescriptor_t) desc, size);
            // return cnnlGetRandomSampleWorkspaceSize((RandomSampleCnnlDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetRandomSampleWorkspaceSize((RandomSampleAclnnDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaGetRandomSampleWorkspaceSize((RandomSampleMusaDescriptor_t) desc, size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopRandomSample(infiniopRandomSampleDescriptor_t desc,
                                          void *workspace,
                                          uint64_t workspace_size,
                                          void *result,
                                          void const *probs,
                                          float random_val,
                                          float topp,
                                          int topk,
                                          float temperature,
                                          void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuRandomSample((RandomSampleCpuDescriptor_t) desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaRandomSample((RandomSampleCudaDescriptor_t) desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangRandomSample((RandomSampleBangDescriptor_t) desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRandomSample((RandomSampleAclnnDescriptor_t) desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu:
            return musaRandomSample((RandomSampleMusaDescriptor_t) desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyRandomSampleDescriptor(infiniopRandomSampleDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyRandomSampleDescriptor((RandomSampleCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroyRandomSampleDescriptor((RandomSampleCudaDescriptor_t) desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyRandomSampleDescriptor((RandomSampleBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRandomSampleDescriptor((RandomSampleAclnnDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu:
            return musaDestroyRandomSampleDescriptor((RandomSampleMusaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}
