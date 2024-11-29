#include "rms_norm_musa.h"
#include "../../utils.h"
#include "../../../devices/musa/common_musa.h"

infiniopStatus_t musaCreateRMSNormDescriptor(MusaHandle_t handle, RMSNormMusaDescriptor_t *desc_ptr,
                                    infiniopTensorDescriptor_t y_desc,
                                    infiniopTensorDescriptor_t x_desc,
                                    infiniopTensorDescriptor_t w_desc,
                                    float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         d = y_desc->shape[1];

    if (x_desc->shape[0] != n || x_desc->shape[1] != d || w_desc->shape[0] != d) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    unsigned long int stride_y = y_desc->strides[0];
    unsigned long int stride_x = x_desc->strides[0];
    auto w_datatype = w_desc->dt;
    *desc_ptr = new RMSNormMusaDescriptor{
        handle->device,
        handle->device_id,
        y_desc->dt,
        n,
        d,
        stride_y,
        stride_x,
        w_datatype,
        epsilon};

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetRMSNormWorkspaceSize(RMSNormMusaDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyRMSNormDescriptor(RMSNormMusaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
