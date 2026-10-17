#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "matrix.h"
#include "dtype.h"

void compute_c_strides(const size_t* shape, u8 ndim, size_t esize, size_t* strides){
    // For contiguous strides calculation
    size_t temp = esize;
    for(int i = (int)ndim - 1; i >= 0; i--){
        // Here use (int)ndim in order to avoid the situation tha ndim == 0 
        strides[i] = temp;
        temp *= shape[i];
    }
}


array ndarray(void* data, const size_t* shape, const size_t* strides, u8 ndim, u8 dtype, bool owns_data){
    /**
    * @brief This function is used to initialize a numpy array variable
    * @param data: is a pointer that points to the address where the array exist
    * @param shape: the shape in logic of the array, since the dimension of data is 1
    * @param ndim: the number of dimensions of the array
    * @param dtype: what kind of type the element is in the array, data-type
    */
    

    size_t esize = dtype_sizeof(dtype);
    if(esize == 0)return array_null();

    u8 n = ndim ? ndim : 1;
    size_t* shape_copy = (size_t*)malloc(n*sizeof(size_t));
    size_t* strides_copy = (size_t*)malloc(n*sizeof(size_t));
    
    // Inspect if the allocation fails
    if(!shape_copy || !strides_copy){
        free(shape_copy); free(strides_copy);
        return array_null();
    }

    
    // Copy shape and strides(if exists)
    memcpy(shape_copy, shape, n*sizeof(size_t));
    for(int i = 0; i < ndim; i++) shape_copy[i] = shape[i];

    if(strides) memcpy(strides_copy, strides, n*sizeof(size_t));
    else compute_c_strides(shape, ndim, esize, strides_copy);

    array creation;
    creation.dtype = dtype;
    creation.data = data;
    creation.ndim = n;
    creation.shape = shape_copy;
    creation.strides = strides_copy;
    creation.owns_data = owns_data;

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

array array_null(){
    array null = {0};
    return null;
}

array zeros(size_t* shape, u8 ndim, dtype_t dtype){
    // The most difficult part is to create a data area that match the size of shape
    int memoryLength = 1;
    for(int i=0; i<ndim; i++){
        memoryLength *= shape[i];
    }

    void* data = calloc(
     memoryLength,
        dtype_sizeof(dtype)
    );

    if(!data) return array_null();

    array a = ndarray(data, shape, NULL, ndim, dtype, true);
    if(!a.shape){
        free(data);
        return array_null();
    }

    return a;
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
    if(!data) return array_null();

    array a = ndarray(data, shape, NULL, ndim, DTYPE_INT32, true);
    if(!a.shape){
        free(data);
        return array_null();
    }

    return a;
}


static inline size_t elem_offset(const array *A, size_t linear_index){
    /*
     * @brief This is a function that can inversely derive the correct index
     *        when A's logic shape matches the linaer memory
     * 
     * @param A, the pointer of an array
     * @param linear_index, if A's logic shape matches the contiguous memory, the correct index
     * */
    size_t off = 0;

    // Prepare for the information we need
    size_t esize = dtype_sizeof(A->dtype);

    for(int i = A->ndim-1; i >= 0; i--){
        size_t coordinate = linear_index % A->shape[i];
        off += coordinate * (A->strides[i] / esize);
        linear_index /= A->shape[i];
    }

    return off;
}

void array_free(array* a){
    if(a->owns_data) free(a->data);
    free(a->shape);
    free(a->strides);
    *a = array_null();
}

int broadcast_shape(const array* A, const array* B, size_t* outShape, u8* outNdim){
    /*
    * @brief 
    * This function is used to calculate the broadcast shape 
    * based on the shapes of A and B.
    * 
    * Every dim of A can match the correpsond dim of B if
    *   1. dimA == dimB
    *   2. dimA == 1 || dimB == 1
    *   3. dimA or dimB do not exists
    * If match, we select the bigger and existing one as the final result dim
    *
    * For example we have
    * A with shape (2,3)
    * B with shape (2,1)
    *
    * For this example, we can compare from end to start 
    *   1. dimA == 3 vs dimB == 1, match, select the bigger one 3
    *   2. dimB == 2 vs dimB == 2, match, select 2
    * So the result outShape is [2,3]
    *
    * @return
    * 0 on success, -1 if shapes are incompatible
    */

    // Use the longer ndim as the output ndim
    u8 ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;
    for(int k=0; k<ndim; k++){
        // k counts from the right
        // k = 0 means the last dimension
        size_t dimA = (k >= A->ndim) ? 1 : A->shape[A->ndim-k-1];
        size_t dimB = (k >= B->ndim) ? 1 : B->shape[B->ndim-k-1];
        if(dimA==dimB || dimA==1 || dimB==1)
            outShape[ndim - k - 1] = (dimA > dimB) ? dimA : dimB;
        else return -1;
    }

    *outNdim = ndim;
    return 0;
}

array broadcast_to(const array* A, const size_t* tshape, u8 tndim){
    /*
     * @brief
     * This function is used to apply the broadcast shape to the specific array, 
     *
     *
     * @param A, the array which needs to applied the broadcast shape
     * @param tshape, the target shape
     * @param tndim, the target number of dims
     *
     * @notes
     * the core step is to calculate the new strides
     *
     * For example we have
     * A = [[1,2,3],
     *      [4,5,6]]
     * B = [[1],
     *      [2]]
     * To match the broadcast shape (2,3), we need to expand B
     * B' = [[1,1,1],
     *       [2,2,2]]
     * The subtle method to "expand" B without copy data is to use the function of strides
     *
     * The memory of B reveal of [1 2], orginal B-strides is (1,1) (Ignore size, pure number)
     * For any index (i,j), we need to let B[0, j] = 1 and of course B[1, j] = 2
     * 
     * So the core is that when we set a certain value to i, the linear index of memory don't
     * change as j changes. 
     * Then => strides[j] = 0 which means the effect of j is 0
     *
     * So, from this small example, the rule is decided as:
     *   1. if a dim have no effect to linear index ==> new_strides[dim] = 0
     *          This happens when (1) strides[dim] == 1 && target_shape[dim] == 0
     *                            (2) strides[dim] does not exist
     *                            both means that we need to "copy" along this axis
     *                            yes, this dim is ineffective
     *   2. otherwise, ==> new_strides[dim] = strides[dim]
     *          This means that along this axis, the linear memory needs to change index
     *          or, this dim is effective
     * */
    // allocate, sometimes the strides and shapes will change
    size_t new_strides[MAX_DIM];

    for(int k=0; k<tndim; k++){
        int dim_t = tndim - 1 - k;
        int dim_a = A->ndim - 1 - k;

        // k counts from the right
        if(dim_a < 0){
            // k >= A->dim
            // which means this dimension dosen't exist in A
            new_strides[dim_t] = 0;
        }else if(A->shape[dim_a] == 1 && tshape[dim_t] > 1){
            new_strides[dim_t] = 0;
        }else{
            // Make A's stride stay still
            new_strides[dim_t] = A->strides[dim_a];
        }
    }

    // return a view, with owns_data flag false
    return ndarray(A->data, tshape, new_strides, tndim, A->dtype, false);
}


#define APPLY_OP(A, B, C, TYPE, LEN, OPERATOR) do{\
            const TYPE* ptr_A = (const TYPE*)(A)->data; \
            const TYPE* ptr_B = (const TYPE*)(B)->data; \
            TYPE* ptr_C = (TYPE*)C->data; \
            for(size_t i = 0; i<(LEN); i++){ \
                TYPE x = ptr_A[elem_offset(A, i)];\
                TYPE y = ptr_B[elem_offset(B, i)];\
                switch(OPERATOR){\
                    case OP_ADD: ptr_C[i] = x + y;break;\
                    case OP_SUB: ptr_C[i] = x - y;break;\
                    case OP_MUL: ptr_C[i] = x * y;break;\
                    case OP_DIV: ptr_C[i] = x / y;break;}\
            }\
}while(0)


array binary_op(const array *A, const array *B, op_t operator_){
    /*
    * @brief
    * This function implement the add mechanism, also implement the 
    * strides mechanism. Created for add() and subtract()
    * The shape of A must match the shape of B, also ndim
    *
    * @param A, the first operand number
    * @param B, the second operand number
    * @param operator, 0 for positive, 1 for negative
    *
    * @notes
    * We have included broadcast mechanism
    * But, when dtype of A and B don't matches each other, will fail.
    * I will add type promotion in the future
    * */

    // dtype
    u8 dtype;
    if(A->dtype != B->dtype)return array_null();
    else {
        dtype = A->dtype;
    }

    // MAX_DIM is defined in matrix.h
    // Which is equivalent to UINT8_MAX
    size_t oshape[MAX_DIM]; u8 ondim; 
    if(broadcast_shape(A, B, oshape, &ondim) != 0) return array_null();

    array Av = broadcast_to(A, oshape, ondim);
    array Bv = broadcast_to(B, oshape, ondim);
    array *pA = &Av;
    array *pB = &Bv;

    size_t total_len = get_length(&Av);

    array out = zeros(oshape, ondim, dtype);
    if(!out.data){
        array_free(&Av); array_free(&Bv);
        return array_null();
    }
    array* pc = &out;

    
    switch (dtype) {
        case DTYPE_INT32: APPLY_OP(pA, pB, pc, int32_t, total_len, operator_);break;
        case DTYPE_INT64: APPLY_OP(pA, pB, pc, int64_t, total_len, operator_);break;
        case DTYPE_FLOAT64: APPLY_OP(pA, pB,  pc, double, total_len, operator_);break;
        case DTYPE_FLOAT32: APPLY_OP(pA, pB,  pc, float, total_len, operator_);break;
        case DTYPE_FLOAT16: APPLY_OP(pA, pB,  pc, _Float16, total_len, operator_);break;
    }

    array_free(&Av); array_free(&Bv);
    return out;
}




array add(array *A, array *B){
    /*
     * @brief calculate the result of A + B
     *
     * @param A, the first operand
     * @param B, the second operand
     * */

    return binary_op(A, B, OP_ADD);

}


array subtract(array *A, array *B){
    /*
     * @brief calculate the result of A - B
     * @param A, the Minuend
     * @param B, the subtrahend 
     * */

    return binary_op(A, B, OP_SUB);
}


array multiply(array *A, array *B){
    /*
     * @brief calculate the result of A * B
     * @param A, the first operand
     * @param B, the second operand
     * */

    return binary_op(A, B, OP_MUL);
}

array divide(array *A, array *B){
    /*
     * @brief calculate the result of A / B
     * @param A, the first operand
     * @param B, the second operand
     * */

    return binary_op(A, B, OP_DIV);
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
    }while(0)

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



#define COPY_DATA(TYPE, NEWDATA, A, totalSize) do{\
        for(size_t index = 0; index < totalSize; index++){\
            size_t orginal_linear_index = elem_offset(A, index);\
            ((TYPE*)NEWDATA)[index] =((TYPE*)A->data)[orginal_linear_index];}\
        }while(0)

void reshape(array* A, size_t* newShape, u8 newDim){
    /*
     * @brief this reshape function is simply implement the condition of
     *        "Copy data when array is not contiguous". Considering 
     *        there are some special conditions that can reshaped without 
     *        change memory layout, I prepare to solve this problem later
     *
     * @param newShape, the new shape array of A
     * @param newDim, cause change shape may lead to increase or reduction
     *                of dimensions
     *
     * @Core The data copy algorithm is a core method of numpy, it perfectly 
     *       match the feature of array with strides/shape properties.
     *       As a contiguous array like [i,j,k], we can directly find the linear
     *       index of the the coordinates --> Use index = shape[1]*shape[2]*i + shape[2]*j + k
     *       So due to the mathematical properties, we can just inversely drive the
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
                COPY_DATA(int64_t, newData, A, totalSize); break;
            case DTYPE_INT32:
                newData = (int32_t*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(int32_t, newData, A, totalSize); break;
            case DTYPE_FLOAT64:
                newData = (double*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(double, newData, A, totalSize); break;
            case DTYPE_FLOAT32:
                newData = calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(float, newData, A, totalSize); break;
            case DTYPE_FLOAT16:
                newData = (_Float16*)calloc(totalSize, dtype_sizeof(A->dtype));
                COPY_DATA(_Float16, newData, A, totalSize); break;
            default: return;
        }
        A->data = newData;
    }
    view(A, newShape, newDim);

}
