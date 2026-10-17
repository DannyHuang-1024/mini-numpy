#ifndef MATRIX_H
#define MATRIX_H

/*  STD libraries Include  */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#define MAX_DIM UINT8_MAX


/*  Self libraries Include */
#include "dtype.h"

typedef uint8_t u8;
typedef uint32_t u32;


typedef struct array{
    void* data;
    u8 ndim;
    size_t* shape;
    size_t* strides;
    dtype_t dtype;
    bool owns_data; // For convenient free memory
}array;

typedef enum {OP_ADD, OP_SUB, OP_DIV, OP_MUL}op_t;

/* ---------- Functions Declaration ----------  */
array ndarray(void* data, const size_t* shape, const size_t* strides, u8 ndim, u8 dtype, bool owns_data);
array zeros(size_t* shape, u8 ndim, dtype_t dtype);
array ones(size_t* shape, u8 ndim);
array array_null();
void array_free(array *A);

void apply_operation_into(const array *A, const array *B, op_t operator_, array *out);

array add(array *A, array *B);
array subtract(array *A, array *B);
array divide(array *A, array *B);
array multiply(array *A, array *B);

// Reductions
void sum(const array* A, void* out);
double mean(const array* A);
void min(const array* A, void* out);
void max(const array* A, void* out);

void view(array* A, size_t* newView, u8 newDim);
void transpose(array* A);
bool is_contiguous(array* A);
void reshape(array* A, size_t* newShape, u8 newDim);
void reshape(array* A, size_t* newShape, u8 newDim);

#endif
