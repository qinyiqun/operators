#include "pooling_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t musaCreatePoolingDescriptor(MusaHandle_t handle,
                                             PoolingMusaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             uint64_t const *kernel_shape,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t n,
                                             int pooling_type) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != n + 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || x->shape[1] != y->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (pooling_type > 1) {
        return STATUS_BAD_PARAM;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    float alpha = 1.0f, beta = 0.0f;

    const auto kernel_ = reinterpret_cast<int const *>(kernel_shape);
    const auto pads_ = reinterpret_cast<int const *>(pads);
    const auto strides_ = reinterpret_cast<int const *>(strides);

    const auto x_shape = reinterpret_cast<int64_t const *>(x->shape);
    const auto x_strides = reinterpret_cast<int64_t const *>(x->strides);
    const auto y_shape = reinterpret_cast<int64_t const *>(y->shape);
    const auto y_strides = reinterpret_cast<int64_t const *>(y->strides);

    musa::dnn::Tensor *x_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *y_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *indices_tensor = new musa::dnn::Tensor();

    if (y->dt == F16) {
        x_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        y_tensor->SetType(musa::dnn::Tensor::Type::HALF);
    } else if (y->dt == F32) {
        x_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        y_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
    }

    x_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    y_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);

    x_tensor->SetNdInfo((int) ndim, x_shape, x_strides);
    y_tensor->SetNdInfo((int) ndim, y_shape, y_strides);

    int *indice = new int[ndim];

    musa::dnn::Pooling* pooling_operator = new musa::dnn::Pooling();
    pooling_operator->SetMode(getPoolingMode(pooling_type));

    int* dilation_ = new int[n];
    std::fill(dilation_, dilation_+((int) n), 1);

    pooling_operator->SetNdInfo((int) n, kernel_, pads_, strides_, (const int*) dilation_);

    *desc_ptr = new PoolingMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        handle->mudnn_handles_t,
        x_tensor,
        y_tensor,
        indices_tensor,
        pooling_operator,
        alpha,
        beta,
    };
    
    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetPoolingWorkspaceSize(PoolingMusaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyPoolingDescriptor(PoolingMusaDescriptor_t desc) {
    delete(desc->x_tensor);
    delete(desc->y_tensor);
    delete(desc->indices_tensor);
    delete(desc->pool_operator);

    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
