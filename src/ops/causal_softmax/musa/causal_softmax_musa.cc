#include "causal_softmax_musa.h"
#include "../../utils.h"
#include "../../../devices/musa/common_musa.h"

infiniopStatus_t musaCreateCausalSoftmaxDescriptor(MusaHandle_t handle,
                                                   CausalSoftmaxMusaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y) {
    unsigned long int ndim = y->ndim;
    // TODO: only support 2d or 3d tensor
    if (ndim != 2 && ndim != 3) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(y->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    unsigned long int total_seq_len = y->shape[ndim - 1];
    unsigned long int seq_len = y->shape[ndim - 2];
    unsigned long int batch_size = 1;
    unsigned long int stride_b = 0;
    unsigned long int stride_i = y->strides[ndim - 2];
    unsigned long int stride_j = y->strides[ndim - 1];
    if (stride_j != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    for (uint64_t i = 0; i < ndim - 2; i++) {
        batch_size *= y->shape[i];
    }
    if (ndim == 3)
        stride_b = y->strides[ndim - 3];
    unsigned int max_items_per_thread = ROUND_UP_DIV(total_seq_len, MAX_THREADS_PER_BLOCK);

    *desc_ptr = new CausalSoftmaxMusaDescriptor{
        handle->device,
        handle->device_id,
        y->dt,
        batch_size,
        stride_b,
        seq_len,
        stride_i,
        total_seq_len,
        stride_j,
        max_items_per_thread};

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetCausalSoftmaxWorkspaceSize(CausalSoftmaxMusaDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyCausalSoftmaxDescriptor(CausalSoftmaxMusaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
