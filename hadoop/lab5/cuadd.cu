#include <stdio.h>

__global__ void add(int *d_out,int *d_in) {
  int i = threadIdx.x;
  
  d_out[i] = d_in[i];
}

int main(int argc, char **argv) {
    const int SIZE = 5;
    const int BYTE = SIZE*sizeof(int);

    int h_in[] = {1,2,3,4,5};
    int h_out[SIZE];


    int *d_in;
    int *d_out;

    cudaMalloc(&d_in, BYTE);
    cudaMalloc(&d_out, BYTE);

    cudaMemcpy((void*)d_in, (void*)h_in,BYTE,cudaMemcpyHostToDevice);
    add<<<1,SIZE>>>(d_out,d_in);
    cudaMemcpy((void*)h_out, (void*)d_out,BYTE,cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < SIZE;i++) {
      printf("%d ",h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}