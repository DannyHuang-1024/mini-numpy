#ifndef DTYPE_H
#define DTYPE_H

#include <stdlib.h>

DTYPE_H// For dtype
// Just need a judgement to discrimate
typedef enum{
    DTYPE_INT32,
    DTYPE_FLOAT64,
    DTYPE_FLOAT32,
    DTYPE_FLOAT16
}dtype_t;

size_t dtype_sizeof(dtype_t dtype);

#endif // !DTYPE_H
