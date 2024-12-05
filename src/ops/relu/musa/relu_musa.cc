#include "relu_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"

infiniopStatus_t musaCreateReluDescriptor(MusaHandle_t handle,
                                          ReluMusaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new ReluMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        ndim,
        data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyReluDescriptor(ReluMusaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
