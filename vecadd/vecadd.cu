/*
@modifier : Nilanka Manoj
@compile : nvcc vecadd.cu -o build/vecadd
@run : ./build/vecadd <<n>>
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

double *a, *b;
double *c, *c2;

__global__ void vecAdd(double *A, double *B, double *C, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   C[i] = A[i] + B[i];
}

void vecAdd_h(double *A1, double *B1, double *C1, double N)
{
   for (int i = 0; i < N; i++)
      C1[i] = A1[i] + B1[i];
}

int main(int argc, char **argv)
{
   if (argc == 2)
   {
      printf("=====================round strating==========================\n");
      int n = atoi(argv[1]);
      int nBytes = n * sizeof(double);
      int block_size, block_no;

      a = (double *)malloc(nBytes);
      b = (double *)malloc(nBytes);
      c = (double *)malloc(nBytes);
      c2 = (double *)malloc(nBytes);

      double *a_d, *b_d, *c_d;

      block_size = 768;
      block_no = (int)ceil(n / block_size) + 1;

      for (int i = 0; i < n; i++)
      {
         a[i] = sin(i) * sin(i);
         b[i] = cos(i) * cos(i);
         c[i] = 0;
         c2[i] = 0;
      }

      printf("Allocating device memory on host..\n");
      cudaMalloc((void **)&a_d, n * sizeof(double));
      cudaMalloc((void **)&b_d, n * sizeof(double));
      cudaMalloc((void **)&c_d, n * sizeof(double));

      printf("Copying to device..\n");
      cudaMemcpy(a_d, a, n * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b, n * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(c_d, c, n * sizeof(double), cudaMemcpyHostToDevice);

      printf("Doing GPU Vector add\n");
      clock_t start_d = clock();
      vecAdd<<<block_no, block_size>>>(a_d, b_d, c_d, n);
      cudaThreadSynchronize();
      clock_t end_d = clock();

      printf("Doing CPU Vector add\n");
      clock_t start_h = clock();
      vecAdd_h(a, b, c2, n);
      clock_t end_h = clock();
      double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;
      double time_h = (double)(end_h - start_h) / CLOCKS_PER_SEC;

      cudaMemcpy(c, c_d, n * sizeof(double), cudaMemcpyDeviceToHost);
      printf("Number of elements: %d GPU Time: %f CPU Time: %f\n", n, time_d, time_h);
      cudaFree(a_d);
      cudaFree(b_d);
      cudaFree(c_d);

      int e = memcmp(c, c2, n);
      printf("compaired error : %d\n", e);
   }
   else
   {
      printf("invalid arguments\n");
   }

   return 0;
}