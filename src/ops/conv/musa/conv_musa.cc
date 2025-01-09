#include "conv_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include <vector>

infiniopStatus_t musaCreateConvDescriptor(MusaHandle_t handle,
                                          ConvMusaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t w,
                                          void const *pads,
                                          void const *strides,
                                          void const *dilations,
                                          uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    const auto new_ndim = std::max(4UL, ndim);
    // convert pads, strides, dilations into int32[]
    int *pad = new int[new_ndim];
    int *stride = new int[new_ndim];
    int *dilation = new int[new_ndim];
    int64_t *x_shape = new int64_t[new_ndim];
    int64_t *w_shape = new int64_t[new_ndim];
    int64_t *y_shape = new int64_t[new_ndim];
    auto pads_ = reinterpret_cast<uint64_t const *>(pads);
    auto strides_ = reinterpret_cast<int64_t const *>(strides);
    auto dilations_ = reinterpret_cast<uint64_t const *>(dilations);
    for (size_t i = 0; i < new_ndim; ++i) {
        pad[i] = i < ndim - 2 ? static_cast<int>(pads_[i]) : 0;
        stride[i] = i < ndim - 2 ? static_cast<int>(strides_[i]) : 1;
        dilation[i] = i < ndim - 2 ? static_cast<int>(dilations_[i]) : 1;
        x_shape[i] = i < ndim ? static_cast<int64_t>(x->shape[i]) : 1;
        w_shape[i] = i < ndim ? static_cast<int64_t>(w->shape[i]) : 1;
        y_shape[i] = i < ndim ? static_cast<int64_t>(y->shape[i]) : 1;
    }

    musa::dnn::Tensor *x_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *y_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *w_tensor = new musa::dnn::Tensor();

    if (y->dt == F16) {
        x_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        y_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        w_tensor->SetType(musa::dnn::Tensor::Type::HALF);
    } else if (y->dt == F32) {
        x_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        y_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        w_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
    }

    x_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    y_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    w_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);

    x_tensor->SetNdInfo((int) new_ndim, x_shape);
    y_tensor->SetNdInfo((int) new_ndim, y_shape);
    w_tensor->SetNdInfo((int) new_ndim, w_shape);

    musa::dnn::Convolution* conv_operator = new musa::dnn::Convolution();
    conv_operator->SetNdInfo((int) new_ndim-2, pad, stride, dilation);
    musa::dnn::Convolution::Algorithm algo = musa::dnn::Convolution::Algorithm::DIRECT;
    size_t workspace_size = 0;

    use_mudnn(handle->mudnn_handles_t, handle->device_id, nullptr, [&](musa::dnn::Handle* handle) {
        printf(" %d \n", conv_operator->GetRecommendForwardAlgorithm(*handle, algo, *y_tensor, *x_tensor, *w_tensor));
        // printf(" %d \n", conv_operator->GetForwardWorkspaceSize(*handle, workspace_size, *y_tensor, *x_tensor, *w_tensor, algo));
    });
    const float alpha = 1.0f;
    const float beta = 0.0f;
    printf("after: %d\n", algo);

    printf("A\n");

    *desc_ptr = new ConvMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        handle->mudnn_handles_t,
        x_tensor,
        w_tensor,
        y_tensor,
        conv_operator,
        algo,
        alpha,
        beta,
        workspace_size};

    delete[] pad;
    delete[] stride;
    delete[] dilation;
    delete[] x_shape;
    delete[] w_shape;
    delete[] y_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetConvWorkspaceSize(ConvMusaDescriptor_t desc, uint64_t *size) {
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyConvDescriptor(ConvMusaDescriptor_t desc) {

    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
