#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>


__device__ float interpolation3D(float currentValue[])
{
    cv::Vec3f low = cv::Vec3b(floor(currentValue[0]), floor(currentValue[1]), floor(currentValue[2]));
    cv::Vec3f high = cv::Vec3b(ceil(currentValue[0]), ceil(currentValue[1]), ceil(currentValue[2]));

    cv::Vec3f c000 = cv::Vec3f(low[0], low[1], high[2]);
    cv::Vec3f c001 = cv::Vec3f(low[0], high[1], high[2]);
    cv::Vec3f c011 = cv::Vec3f(low[0], high[1], low[2]);
    cv::Vec3f c010 = cv::Vec3f(low[0], low[1], low[2]);
    cv::Vec3f c100 = cv::Vec3f(high[0], low[1], high[2]);
    cv::Vec3f c101 = cv::Vec3f(high[0], high[1], high[2]);
    cv::Vec3f c111 = cv::Vec3f(high[0], high[1], low[2]);
    cv::Vec3f c110 = cv::Vec3f(high[0], low[1], low[2]);

    float x_d = (currentValue[0] - low[0]) / (high[0] - low[0]);
    float y_d = (currentValue[1] - low[1]) / (high[1] - low[1]);
    float z_d = (currentValue[2] - low[2]) / (high[2] - low[2]);

    x_d = x_d > 0 ? x_d : 0;
    y_d = y_d > 0 ? y_d : 0;
    z_d = z_d > 0 ? z_d : 0;

    cv::Vec3f c00 = c000 * (1 - x_d) + c100 * x_d;
    cv::Vec3f c01 = c001 * (1 - x_d) + c101 * x_d;
    cv::Vec3f c11 = c010 * (1 - x_d) + c110 * x_d;
    cv::Vec3f c10 = c011 * (1 - x_d) + c111 * x_d;

    cv::Vec3f c0 = c00 * (1 - y_d) + c10 * y_d;
    cv::Vec3f c1 = c01 * (1 - y_d) + c11 * y_d;

    return c0 * (1 - z_d) + c1 * z_d;
}


__global__ void  colorManagement(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> lut[], int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    cv::Vec3b finalValue;

    if (dst_x < cols && dst_y < rows)
    {
        float currentValue[3] = {(float)src(dst_y, dst_x).x, (float)src(dst_y, dst_x).y, (float)src(dst_y, dst_x).z };

        currentValue[0] = currentValue[0] > 63 ? 63 : currentValue[0];
        currentValue[1] = currentValue[1] > 63 ? 63 : currentValue[1];
        currentValue[2] = currentValue[2] > 63 ? 63 : currentValue[2];

        const float finalValue[3] = interpolation3D(currentValue);

        int x = lut[finalValue[2]][finalValue[0], finalValue[1]].x;
        
        dst(dst_y, dst_x).x = (unsigned char)(x);
        dst(dst_y, dst_x).y = (unsigned char)(lut[finalValue[2]][finalValue[0], finalValue[1]][1]);
        dst(dst_y, dst_x).z = (unsigned char)(lut[finalValue[2]][finalValue[0], finalValue[1]][2]);

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