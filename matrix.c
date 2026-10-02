#include <stddef.h>
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

    // Needs to copy
    size_t* shape_copy = malloc(ndim * sizeof(size_t));
    for(int i = 0; i < ndim; i++) shape_copy[i] = shape[i];
    creation.shape = shape_copy;

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


size_t get_length(const array* A){
    /*
     * @brief Return the impratical length of A, the number of elements
     * */
    size_t total_len = 1;
    for(int i=0; i<A->ndim; i++){
        total_len *= A->shape[i];
    }

    return total_len;
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

    size_t total_len = get_length(A);

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
        ARRAY_ADD_LOOP(A, B, out, int32_t*, total_len, operator_cal);
    }else if(dtype == DTYPE_FLOAT64){
        ARRAY_ADD_LOOP(A, B, out, double*, total_len, operator_cal);
    }else if (dtype == DTYPE_FLOAT32){
        ARRAY_ADD_LOOP(A, B, out, float*, total_len, operator_cal);
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

    size_t total_len = get_length(A);

    if(dtype == DTYPE_INT32){
        ARRAY_MULTIPLY_LOOP(A, B, out, int32_t*, total_len, operator_);
    }else if(dtype == DTYPE_FLOAT64){
        ARRAY_MULTIPLY_LOOP(A, B, out, double*, total_len, operator_);
    }else if(dtype == DTYPE_FLOAT32){
        ARRAY_MULTIPLY_LOOP(A, B, out, float*, total_len, operator_);
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


#define SUM_IMPL(A, OUT, TYPE, total_len)\
        TYPE result = 0; \
        TYPE* ptr_A = (TYPE*)A->data;\
        for(int i = 0; i < total_len; i++){\
            result +=  ptr_A[i];}\
        *((TYPE*)OUT) = result;

void sum(const array* A, void* out){
    /*
     * @brief This funciton will calculate the sum of all elements in the matrix
     * 
     * @param A, the pointer of the array need to be process,
     *           It is a const pointer, so this function will not change the address.
     * @param out, the output pointer, we use void* to accept the result, so the 
     *             in the high level API it can just use different types.
     * */

    size_t total_len = get_length(A);

    if(A->dtype == DTYPE_INT32){
        SUM_IMPL(A, out, int32_t, total_len);
    }else if(A->dtype == DTYPE_FLOAT64){
        SUM_IMPL(A, out, double, total_len);
    }else if(A->dtype == DTYPE_FLOAT32){
        SUM_IMPL(A, out, float, total_len);
    }
}


double mean(const array* A){
    /*
     * @brief This function used to calculate the mean value of the array
     *        The implementation calls from sum() and use a DTYPE judgement
     *        to route different type of arrays
     * 
     * @param Only ndarray A
     * */

    double result = 0.0;

    if(A->dtype == DTYPE_INT32){
        int32_t sum_of_element = 0;
        sum(A, &sum_of_element);
        result = (double)sum_of_element;
    }else if(A->dtype == DTYPE_FLOAT64){
        sum(A, &result);
    }else if(A->dtype == DTYPE_FLOAT32){
        float sum_of_element = 0.0f;
        sum(A, &sum_of_element);
        result = (double)sum_of_element;
    }

    size_t total_len = get_length(A);

    return result/total_len;
}


#define FIND_EXTREME(A, LENGTH, TYPE, OUT, CMP)\
    do{\
    TYPE EXTREME= ((TYPE*)A)[0];\
    const TYPE* data = (const TYPE*)(A);\
    for(size_t i=1; i<LENGTH; i++){\
        if(data[i] CMP EXTREME) EXTREME= data[i];}\
    *((TYPE*)OUT) = EXTREME;}\
    while(0)

void min(const array* A, void* out){
    if(A->dtype == DTYPE_INT32){
        FIND_EXTREME(A->data, get_length(A), int32_t, out, <);

    }else if(A->dtype == DTYPE_FLOAT64){
        double minNum;
        FIND_EXTREME(A->data, get_length(A), double, out, <);
    }else if(A->dtype == DTYPE_FLOAT32){
        float minNum;
        FIND_EXTREME(A->data, get_length(A), float, out, <);
    }
}

void max(const array* A, void* out){
    if(A->dtype == DTYPE_INT32){
        FIND_EXTREME(A->data, get_length(A), int32_t, out, >);
    }else if(A->dtype == DTYPE_FLOAT64){
        double minNum;
        FIND_EXTREME(A->data, get_length(A), double, out, >);
    }else if(A->dtype == DTYPE_FLOAT32){
        float minNum;
        FIND_EXTREME(A->data, get_length(A), float, out, >);
    }
}

void view(array* A, size_t* newView, u8 newDim){
    /*
     * @brief reshape the array with wanted strides
     * */

    // Update shape
    for(int i=0; i < newDim; i++){
        A->shape[i] = newView[i];
    }

    size_t stride = dtype_sizeof(A->dtype);
    for(int i=newDim-1; i >= 0; i--){
        A->strides[i] = stride;
        // Very important, the calculation of stride is so cumbersome
        // So please be carefully understand
        stride *= A->shape[i];
    }
    A->ndim = newDim;
}

#define SWAP(A, B) do{\
    __typeof__(*A) tmp = (*A);\
    (*(A)) = (*(B));\
    (*(B)) = tmp;\
    }while(0);

void transpose(array* A){
    int left = 0, right = A->ndim-1;
    while(left < right){
        SWAP(A->shape+left, A->shape+right);
        SWAP(A->strides+left, A->strides+right);
        right--;
        left++;
    }
}




bool is_contiguous(array* A){
    size_t sizeExpected = dtype_sizeof(A->dtype);
    for(int i=A->ndim-1; i >=0; i--){
        if(A->shape[i] == 1){
            // Here is the subtle thing that is a dimension with shape 1
            // no matter how stride be, it is contiguous
            continue;
        }

        if(A->strides[i] != sizeExpected){
            return false;
        }

        sizeExpected *= A->shape[i];
    }
    return true;
}

#define COPY_DATA(TYPE, NEWDATA, A) do{\
        for(size_t index = 0; index < totalSize; index++){\
            size_t idx = index;\
            size_t orginal_linear_index = 0;\
            for(int i = A->ndim - 1; i >= 0; i--){\
                orginal_linear_index += (A->strides[i])/dtype_sizeof(A->dtype) * (idx%(A->shape[i]));\
                idx /= A->shape[i];}\
            ((TYPE*)NEWDATA)[index] =((TYPE*)A->data)[orginal_linear_index];}\
        }while(0);

void reshape(array* A, size_t* newShape, u8 newDim){
    /*
     * @brief this reshape function is simply implement the condition of
     *        "Copy data when array is not contiguous". Considering 
     *        there are some special condition that can reshaped without 
     *        change memory layout, I prepare to solve this problem
     *
     * @Core The data copy algorithm is a core method of numpy, it perfectly 
     *       match the feature of array with strides/shape properties.
     *       As a contiguous array like [i,j,k], we can directly find the linear
     *       index of the the coordinates --> Use index = shape[1]*shape[2]*i + shape[2]*j + k
     *       So due to the mathematical properties, we can just inversely calculate the
     *       array coordinates by linear index --> 
     *          1. k = index % shape[2] 
     *          2. new_index = index/shape[2] = shape[1]*i + j, here again we get the simliar form
     *          3. j = new_index % shape[1]
     *          4. new_new_index = new_index/shape[1] = i
     *          5. i = new_new_index % shape[0], this is because i is restricted at the range of [0, shape[0]-1]
     *          6. i,j,k is generated
     *      And then, once we know the corresponding array coordinates of contiguous linear, we then
     *      should know what exactly this index reflects to orginal non contiguous array, this is easy,
     *      use strides then.
     * */

    // Check element counts match
    size_t newTotal = 1;
    for(int i = 0; i < newDim; i++) newTotal *= newShape[i];
    size_t totalSize = get_length(A);
    if(newTotal != totalSize) return;

    if(!is_contiguous(A)){
        // Non contiguous
        void* newData;
        switch(A->dtype){
            case DTYPE_INT64:
                newData = (int64_t*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(int64_t, newData, A); break;
            case DTYPE_INT32:
                newData = (int32_t*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(int32_t, newData, A); break;
            case DTYPE_FLOAT64:
                newData = (double*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(double, newData, A); break;
            case DTYPE_FLOAT32:
                newData = calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(float, newData, A); break;
            case DTYPE_FLOAT16:
                newData = (_Float16*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(_Float16, newData, A);
            default: return;
        }
        A->data = newData;
    }
    view(A, newShape, newDim);

}
