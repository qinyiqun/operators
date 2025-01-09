#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include "conv_musa.h"

infiniopStatus_t conv_mt_gpu(ConvMusaDescriptor_t desc, void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void const *w, void *stream) {
    checkMusaError(musaSetDevice(desc->device_id));
    desc->y_tensor->SetAddr(y);
    desc->x_tensor->SetAddr(x);
    desc->w_tensor->SetAddr(w);
    printf("b\n");

    use_mudnn(desc->mudnn_handles_t, desc->device_id, (musaStream_t) stream, [&](musa::dnn::Handle* handle) {
        desc->conv_operator->Run(*handle, *(desc->y_tensor), *(desc->x_tensor), *(desc->w_tensor), desc->algo, nullptr);

    });
    return STATUS_SUCCESS;
}

infiniopStatus_t musaConv(ConvMusaDescriptor_t desc,
                          void *workspace, uint64_t workspace_size,
                          void *y, void const *x, void const *w,
                          void *stream) {
    if (desc->dtype == F16 || desc->dtype == F32) {
        return conv_mt_gpu(desc, workspace, workspace_size, y, x, w, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
