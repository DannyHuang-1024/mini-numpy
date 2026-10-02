#ifndef MATRIX_H
#define MATRIX_H

/*  STD libraries Include  */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#define MAXLENGTH 100


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
}array;


/* ---------- Functions Declaration ----------  */
array ndarray(void* data, size_t* shape, u8 ndim, u8 dtype);
void add_impl(array *A, array *B, bool operator_, array *out);
void add(array *A, array *B, array *out);
void subtract(array *A, array *B, array *out);
void divide(array *A, array *B, array *out);
void multiply(array *A, array *B, array *out);

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
