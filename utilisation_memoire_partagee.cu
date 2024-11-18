%%writefile utilisation_memoire_partagee.cu

// étape 3 

#include <iostream>

// Taille des matrices
#define N 1024

// Noyau CUDA avec mémoire partagée
__global__ void matrixMulSharedMemory(const float *A, const float *B, float *C, int n) {
    __shared__ float A_shared[32][32];
    __shared__ float B_shared[32][32];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0;

    for (int i = 0; i < (n + 31) / 32; ++i) {
        if (row < n && (i * 32 + tx) < n)
            A_shared[ty][tx] = A[row * n + i * 32 + tx];
        else
            A_shared[ty][tx] = 0.0;

        if (col < n && (i * 32 + ty) < n)
            B_shared[ty][tx] = B[(i * 32 + ty) * n + col];
        else
            B_shared[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

void initializeMatrix(float *mat, int n, float value) {
    for (int i = 0; i < n * n; ++i) {
        mat[i] = value;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;  // Matrices sur l'hôte
    float *d_A, *d_B, *d_C;  // Matrices sur le GPU

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    initializeMatrix(h_A, N, 1.0f);
    initializeMatrix(h_B, N, 2.0f);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matrixMulSharedMemory<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Résultat (partie de la matrice C) :" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}