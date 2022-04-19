#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

 void printArr( int arr[], int n )
{
    int i;
    for ( i = 0; i < n; ++i )
        printf( "%d ", arr[i] );
}
__device__ int d_size;

__global__ void partition (int *arr, int *arr_l, int *arr_h, int n)
{
    int z = blockIdx.x*blockDim.x+threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z<n)
      {
        int h = arr_h[z];
        int l = arr_l[z];
        int x = arr[h];
        int i = (l - 1);
        int temp;
        for (int j = l; j <= h- 1; j++)
          {
            if (arr[j] <= x)
              {
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
          }
        temp = arr[i+1];
        arr[i+1] = arr[h];
        arr[h] = temp;
        int p = (i + 1);
        if (p-1 > l)
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p-1;  
          }
        if ( p+1 < h )
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p+1;
            arr_h[ind] = h; 
          }
      }
}
 
void quickSortIterative (int arr[], int l, int h)
{
    int *lstack, *hstack;
    lstack = (int *)malloc(sizeof(int)*(h-l+1));
    hstack = (int *)malloc(sizeof(int)*(h-l+1));
    // int hstack[ h - l + 1];
 
    int top = -1, *d_d, *d_l, *d_h;
 
    lstack[ ++top ] = l;
    hstack[ top ] = h;

    cudaMalloc(&d_d, (h-l+1)*sizeof(int));
    cudaMemcpy(d_d, arr,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h-l+1)*sizeof(int));
    cudaMemcpy(d_l, lstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h-l+1)*sizeof(int));
    cudaMemcpy(d_h, hstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);
    int n_t = 1;
    int n_b = 1;
    int n_i = 1; 
    while ( n_i > 0 )
    {
        partition<<<n_b,n_t>>>( d_d, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost); 
        if (answer < 1024)
          {
            n_t = answer;
          }
        else
          {
            n_t = 1024;
            n_b = answer/n_t + (answer%n_t==0?0:1);
          }
        n_i = answer;
        cudaMemcpy(arr, d_d,(h-l+1)*sizeof(int),cudaMemcpyDeviceToHost);
    }
}
 
 
// A utility function to swap two elements
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
 
/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partitionSeq(int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element and indicates the right position of pivot found so far
 
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
 
/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
void quickSortSeq(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partitionSeq(arr, low, high);
 
        // Separately sort elements before
        // partition and after partition
        quickSortSeq(arr, low, pi - 1);
        quickSortSeq(arr, pi + 1, high);
    }
}
 
 

 
int main()
{
    // int arr[5000];
    srand(time(NULL));
    // for (int i = 0; i<5000; i++)
    //    {
    //      arr[i] = rand ()%10000;
    //    }
    // int n = sizeof( arr ) / sizeof( *arr );
    // quickSortIterative( arr, 0, n - 1 );
    // printArr( arr, n );
    // return 0;

    clock_t start, end;
    double time_used;
    int n_list[] = {10, 100, 1000, 10000, 100000, 1000000};
    int i, j;
    for(j=0; j<6; j++){
      int *arr = (int *)malloc(sizeof(int)*n_list[j]);
      for (i = 0; i < n_list[j]; i++)
         {
           arr[i] = rand ()%10000;
         }
        printf("############ LENGTH OF LIST: %d ############\n", n_list[j]);
        start = clock();
        quickSortIterative( arr, 0, n_list[j] - 1 );
        end = clock();
        time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("TIME TAKEN(Parallel GPU): %f\n", time_used);


        start = clock();
        quickSortSeq( arr, 0, n_list[j] - 1 );
        end = clock();
        time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("TIME TAKEN(Sequential CPU): %f\n", time_used);

        printf("##################################################\n");
    }

}