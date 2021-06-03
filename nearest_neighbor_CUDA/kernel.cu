#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>


__device__ int3 nn_interpolation(float3 currentValue)
{
    float3 low = { floor((double)currentValue.x), floor((double)currentValue.y), floor((double)currentValue.z) };
    float3 high = { ceil((double)currentValue.x), ceil((double)currentValue.y), ceil((double)currentValue.z) };

    float x_d = (currentValue.x - low.x) / (high.x - low.x);
    float y_d = (currentValue.y - low.y) / (high.y - low.y);
    float z_d = (currentValue.z - low.z) / (high.z - low.z);

    x_d = x_d > 0 ? x_d : 0;
    y_d = y_d > 0 ? y_d : 0;
    z_d = z_d > 0 ? z_d : 0;

    float3 c;

    c.x = x_d < 0.5 ? low.x : high.x;
    c.y = y_d < 0.5 ? low.y : high.y;

    c.z = z_d < 0.5 ? low.z : high.z;

    return { (int)(c.x), (int)(c.y), (int)(c.z) };
}


__global__ void  colorManagement(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> lut, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int3 finalValue;

    if (dst_x < cols&& dst_y < rows)
    {
        float3 currentValue = { (float)src(dst_y, dst_x).x / 4.0, (float)src(dst_y, dst_x).y / 4.0, (float)src(dst_y, dst_x).z / 4.0 };

        currentValue.x = currentValue.x > 63 ? 63 : currentValue.x;
        currentValue.y = currentValue.y > 63 ? 63 : currentValue.y;
        currentValue.z = currentValue.z > 63 ? 63 : currentValue.z;

        finalValue = nn_interpolation(currentValue);

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