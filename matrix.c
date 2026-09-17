#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "matrix.h"
#include "dtype.h"


array ndarray(void* data, size_t* shape, u8 ndim, u8 dtype){
    /**
    * @brief This function is used to initialize a numpy array variable
    * @param data: is a pointer that points to the address where the array exist
    * @param shape: the shape in logic of the array, since the dimension of data is 1
    * @param ndim: the number of dimensions of the array
    * @param dtype: what kind of type the element is in the array, data-type
    */
    
    array creation;
    creation.dtype = dtype;
    creation.data = data;
    creation.ndim = ndim;
    creation.shape = shape;
    size_t* stride = (size_t*)malloc(ndim*sizeof(size_t));
    int temp = 1;
    for(int i= 0; i < ndim; i++){
        temp *= shape[i];
    }

    for(int i=0; i < ndim; i++){
        temp /= shape[i];
        // Use dtype_sizeof funciton to get size_t base on different dtype
        stride[i] = temp * dtype_sizeof(dtype);
    }

    creation.strides = stride;

    return creation;
}

array zeros(size_t* shape, u8 ndim, dtype_t dtype){
    // The most difficult part is to create a data area that match the size of shape
    int memoryLength = 1;
    for(int i=0; i<ndim; i++){
        memoryLength *= shape[i];
    }

    // Considering the later modification base on special demands, 
    // here we use two macro to represent the type and dtype
    void* data = calloc(
     memoryLength,
        dtype_sizeof(dtype)
    );

    return ndarray(data, shape, ndim, dtype);
};


array ones(size_t* shape, u8 ndim){
    int memoryLength = 1;
    for(int i=0; i<ndim; i++){
        memoryLength *= shape[i];
    }

    int32_t* data = (int32_t*)malloc(memoryLength*sizeof(int32_t));
    for(int i=0; i<memoryLength; i++){
        data[i] = 1;
    }

    return ndarray(data, shape, ndim, DTYPE_INT32);
}


#define ARRAY_ADD_LOOP(A, B, C, TYPE, LEN, OPERATOR) \
            TYPE (ptr_A) = (TYPE)A->data; \
            TYPE (ptr_B) = (TYPE)B->data; \
            TYPE (ptr_C) = (TYPE)out->data; \
            for(int i = 0; i<(LEN); i++){ \
                ptr_C[i] = ptr_A[i] + (OPERATOR)*(ptr_B[i]);}

#define ARRAY_MULTIPLY_LOOP(A, B, C, TYPE, LEN, OPERATOR) \
            TYPE (ptr_A) = (TYPE)A->data;\
            TYPE (ptr_B) = (TYPE)B->data;\
            TYPE (ptr_C) = (TYPE)C->data;\
            for(int i = 0; i<(LEN); i++){\
                if(OPERATOR == 0){\
                    ptr_C[i] = ptr_A[i] * (ptr_B[i]);}\
                else ptr_C[i] =(ptr_A[i]) / (ptr_B[i]);}



void add_impl(array *A, array *B, bool operator_, array *out){
    /*
    * @brief
    * This function implement the add mechanism
    * Just created for add() and subtract()
    * The shape of A must match the shape of B, also ndim
    *
    * @param A, the first operand number
    * @param B, the second operand number
    * @param operator, 0 for positive, 1 for negative
    *
    * @notes
    * And we temporarily haven't applied the broadcast mechanism
    * So A has the same shape of B
    * */

    // Shape inspection
    if(A->ndim != B->ndim)return;
    
    for(int i=0; i<A->ndim; i++){
        if(A->shape[i] != B->shape[i]) return;
    }

    // Calculate the length of array
    size_t total_len = 1;
    for(int i=0; i<A->ndim; i++){
        total_len *= A->shape[i];
    }

    // If the dtypes of two are unequal
    u8 dtype;
    if(A->dtype != B->dtype)return;
    else {
        dtype = A->dtype;
    }
    
    int operator_cal;
    if(operator_ == 0){
        // add signal
        operator_cal = 1;
    }else{
        operator_cal = -1;
    }

    if(dtype == DTYPE_INT32){
        ARRAY_ADD_LOOP(A, B, out, int32_t*, total_len, operator_cal)
    }else if(dtype == DTYPE_FLOAT64){
        ARRAY_ADD_LOOP(A, B, out, double*, total_len, operator_cal)
    }

}

void mutiply_impl(array* A, array* B, bool operator_, array *out){
    /*
     * @brief this funciton is used to implement multiplication and division
     *
     * @param A, the first operand
     * @param B, the second operand
     *
     * @notes
     * No broadcast mechanism
     * */

    // Shape inspection
    if(A->ndim != B->ndim)return;
    
    for(int i=0; i<A->ndim; i++){
        if(A->shape[i] != B->shape[i]) return;
    }

    // If the dtypes of two are unequal
    u8 dtype;
    if(A->dtype != B->dtype)return;
    else {
        dtype = A->dtype;
    }

    // Calculate the length of array
    size_t total_len = 1;
    for(int i=0; i<A->ndim; i++){
        total_len *= A->shape[i];
    }

    if(dtype == DTYPE_INT32){
        ARRAY_MULTIPLY_LOOP(A, B, out, int32_t*, total_len, operator_);
    }else if(dtype == DTYPE_FLOAT64){
        ARRAY_MULTIPLY_LOOP(A, B, out, double*, total_len, operator_);
    }

}

void add(array *A, array *B, array *out){
    /*
     * @brief calculate the result of A + B
     *
     * @param A, the first operand
     * @param B, the second operand
     * */

    add_impl(A, B, 0, out);

}


void subtract(array *A, array *B, array *out){
    /*
     * @brief calculate the result of A - B
     * @param A, the Minuend
     * @param B, the subtrahend 
     * */

    add_impl(A, B, 1, out);
}


void multiply(array *A, array *B, array *out){
    /*
     * @brief calculate the result of A * B
     * @param A, the first operand
     * @param B, the second operand
     * */

    mutiply_impl(A, B, 0, out);
}

void divide(array *A, array *B, array *out){
    /*
     * @brief calculate the result of A / B
     * @param A, the first operand
     * @param B, the second operand
     * */

    mutiply_impl(A, B, 1, out);
}
