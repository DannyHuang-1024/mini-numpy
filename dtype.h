#ifndef DTYPE_H
#define DTYPE_H

#include <stdlib.h>

// For dtype
typedef enum{
    DTYPE_INT64,
    DTYPE_INT32,
    DTYPE_FLOAT64,
    DTYPE_FLOAT32,
    DTYPE_FLOAT16
}dtype_t;

size_t dtype_sizeof(dtype_t dtype);

#endif // !DTYPE_H
