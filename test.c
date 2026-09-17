#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

void printArray(array a){
}

int main(){
    u32 matrix1[][3] = {{1,2,3},{4,5,6}};
    u32 matrix2[][3] = {{2,4,6},{8,10,12}};
    size_t shape[] = {2,3};
    array a = ndarray(matrix1,  shape, 2, DTYPE_INT32);
    array b = ndarray(matrix2,  shape, 2, DTYPE_INT32);

    array c = ndarray(NULL, shape, 2, DTYPE_INT32);
    c.data = (int32_t*)malloc(6*sizeof(int32_t));
    // subtract(&a, &b, &c);
    // divide(&b, &a, &c);
    multiply(&b, &a, &c);
    subtract(&c, &b, &c);

    for(int i=0; i<6; i++){
        printf("%d ", ((int32_t*)c.data)[i]);
    }
    printf("\n");

    return 0;
}

