#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>


__device__ int3 interpolation3D(float3 currentValue)
{
    float3 low = { floor((double)currentValue.x), floor((double)currentValue.y), floor((double)currentValue.z) };
    float3 high = { ceil((double)currentValue.x), ceil((double)currentValue.y), ceil((double)currentValue.z) };
         
    float3 c000 = { low.x, low.y, high.z };
    float3 c001 = { low.x, high.y, high.z };
    float3 c011 = { low.x, high.y, low.z };
    float3 c010 = { low.x, low.y, low.z };
    float3 c100 = { high.x, low.y, high.z };
    float3 c101 = { high.x, high.y, high.z };
    float3 c111 = { high.x, high.y, low.z };
    float3 c110 = { high.x, low.y, low.z };

    float x_d = (currentValue.x - low.x) / (high.x - low.x);
    float y_d = (currentValue.y - low.y) / (high.y - low.y);
    float z_d = (currentValue.z - low.z) / (high.z - low.z);

    x_d = x_d > 0 ? x_d : 0;
    y_d = y_d > 0 ? y_d : 0;
    z_d = z_d > 0 ? z_d : 0;

    float3 c00 = {c000.x * (1 - x_d) + c100.x * x_d, c000.y * (1 - x_d) + c100.y * x_d , c000.z * (1 - x_d) + c100.z * x_d };
    float3 c01 = {c001.x * (1 - x_d) + c101.x * x_d, c001.y * (1 - x_d) + c101.y * x_d , c001.z * (1 - x_d) + c101.z * x_d };
    float3 c11 = {c010.x * (1 - x_d) + c110.x * x_d, c010.y * (1 - x_d) + c110.y * x_d , c010.z * (1 - x_d) + c110.z * x_d };
    float3 c10 = {c011.x * (1 - x_d) + c111.x * x_d, c010.y * (1 - x_d) + c110.y * x_d , c010.z * (1 - x_d) + c110.z * x_d };
    
    float3 c0 = { c00.x * (1 - y_d) + c10.x * y_d, c00.y * (1 - y_d) + c10.y * y_d, c00.z * (1 - y_d) + c10.z * y_d };
    float3 c1 = { c01.x * (1 - y_d) + c11.x * y_d, c01.y * (1 - y_d) + c11.y * y_d, c01.z * (1 - y_d) + c11.z * y_d };

    return { (int)(c0.x * (1 - z_d) + c1.x * z_d), (int)(c0.y * (1 - z_d) + c1.y * z_d), (int)(c0.z * (1 - z_d) + c1.z * z_d) };
}


__global__ void  colorManagement(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> lut, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int3 finalValue;

    if (dst_x < cols && dst_y < rows)
    {
        float3 currentValue = {(float)src(dst_y, dst_x).x / 4.0, (float)src(dst_y, dst_x).y / 4.0, (float)src(dst_y, dst_x).z / 4.0 };

        currentValue.x = currentValue.x > 63 ? 63 : currentValue.x;
        currentValue.y = currentValue.y > 63 ? 63 : currentValue.y;
        currentValue.z = currentValue.z > 63 ? 63 : currentValue.z;

        finalValue = interpolation3D(currentValue);
        
        dst(dst_y, dst_x) = lut(finalValue.z, finalValue.y * 64 + finalValue.x);

    }


}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


void colorManagementCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& lut, int dimX, int dimY)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    colorManagement << <grid, block >> > (src, dst, lut, dst.rows, dst.cols);
}