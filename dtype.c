#include <stdlib.h>

#include "dtype.h" 

size_t dtype_sizeof(dtype_t dtype){
    switch(dtype){
        case DTYPE_FLOAT64:
            return sizeof(double);
        case DTYPE_FLOAT32:
            return sizeof(float);
        case DTYPE_FLOAT16:
            return sizeof(_Float16);
        case DTYPE_INT32:
            return sizeof(int32_t);
        default:
            return 0;
    }
}
