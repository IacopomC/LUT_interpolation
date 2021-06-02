#include <opencv2/opencv.hpp>
#include <math.h>

cv::Vec3b interpolation1D(cv::Vec3f currentValue, cv::Vec3f low, cv::Vec3f high)
{
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


void colorManagement(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b> lut[])
{

    dst.create(src.rows, src.cols);
    dst = cv::Vec3b(0, 0, 0);

    cv::Vec3f currentValue;
    cv::Vec3f low;
    cv::Vec3f high;

    cv::Vec3b finalValue;

    #pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {

            currentValue = (cv::Vec3f) src.at<cv::Vec3b>(i, j) / 4;

            currentValue[0] = currentValue[0] > 63 ? 63 : currentValue[0];
            currentValue[1] = currentValue[1] > 63 ? 63 : currentValue[1];
            currentValue[2] = currentValue[2] > 63 ? 63 : currentValue[2];

            low = cv::Vec3b(floor(currentValue[0]), floor(currentValue[1]), floor(currentValue[2]));
            high = cv::Vec3b(ceil(currentValue[0]), ceil(currentValue[1]), ceil(currentValue[2]));

            finalValue = interpolation1D(currentValue, low, high);

            dst.at<cv::Vec3b>(i, j) = lut[finalValue[2]].at<cv::Vec3b>(finalValue[0], finalValue[1]);
        }
    }
}
