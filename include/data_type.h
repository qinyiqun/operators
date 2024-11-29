﻿#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

typedef struct DataLayout {
    unsigned short
        packed : 8,
        sign : 1,
        size : 7,
        mantissa : 8,
        exponent : 8;

    bool operator==(const DataLayout &other) const {
        union TypePun {
            DataLayout layout;
            unsigned int i;
        } pun;
        pun.layout = *this;
        auto a_ = pun.i;
        pun.layout = other;
        auto b_ = pun.i;
        return a_ == b_;
    }

    bool operator!=(const DataLayout &other) const {
        return !(*this == other);
    }
} DataLayout;

typedef struct DataLayout DT;

// clang-format off
constexpr static struct DataLayout
    I8   = {1, 1, 1,  7,  0},
    I16  = {1, 1, 2, 15,  0},
    I32  = {1, 1, 4, 31,  0},
    I64  = {1, 1, 8, 63,  0},
    U8   = {1, 0, 1,  8,  0},
    U16  = {1, 0, 2, 16,  0},
    U32  = {1, 0, 4, 32,  0},
    U64  = {1, 0, 8, 64,  0},
    F16  = {1, 1, 2, 10,  5},
    BF16 = {1, 1, 2,  7,  8},
    F32  = {1, 1, 4, 23,  8},
    F64  = {1, 1, 8, 52, 11};
// clang-format on

DT get_F16();

DT get_U32();

DT get_F32();

DT get_U64();

#endif// __DATA_TYPE_H__
