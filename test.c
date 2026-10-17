#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dtype.h"
#include "matrix.h"
void broadcastTest();
void transposeTest();

void printArray(array a){
}

int main(){
    transposeTest();
    broadcastTest();


    return 0;
}
void transposeTest(){
    printf("\nTransposed And Reshape Test\n");
    int32_t matrix1[][3] = {{1,2,3},{4,5,6}};
    int32_t matrix2[][1] = {{2},{8}};
    size_t shape1[] = {2,3};
    size_t shape2[] = {2,1};

    array a = ndarray(matrix1,  shape1, NULL,  2, DTYPE_INT32, true);
    array b = ndarray(matrix2,  shape2, NULL, 2, DTYPE_INT32, true);
    array c = zeros(shape1, 2, DTYPE_INT32);

    transpose(&a);
    size_t new_shape[2] = {3,2};
    for(int i=0; i<6; i++){
        printf("%d ", ((int32_t*)a.data)[i]);
    }
    printf("\n");
    reshape(&a, new_shape, 2);
    for(int i=0; i<6; i++){
        printf("%d ", ((int32_t*)a.data)[i]);
    }
    printf("\n");
    reshape(&a, new_shape, 2);
}

void broadcastTest(){
    printf("\nBroadcast Test\n");
    int32_t matrix1[][3] = {{1,2,3},{4,5,6}};
    int32_t matrix2[][1] = {{2},{8}};
    size_t shape1[] = {2,3};
    size_t shape2[] = {2,1};

    array a = ndarray(matrix1,  shape1, NULL, 2, DTYPE_INT32, false);
    array b = ndarray(matrix2,  shape2, NULL, 2, DTYPE_INT32, false);

    array c = add(&a, &b);
    
    if(!c.strides) printf("yes\n");
    for(int i=0; i<6; i++){
        printf("%d ", ((int32_t*)c.data)[i]);
    }
    array_free(&c);

    printf("\n");
}

