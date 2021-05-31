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

    //std::cout << "c000 " << c000 << std::endl;
    //std::cout << "c001 " << c001 << std::endl;
    //std::cout << "c011 " << c011 << std::endl;
    //std::cout << "c010 " << c010 << std::endl;
    //std::cout << "c100 " << c100 << std::endl;
    //std::cout << "c101 " << c101 << std::endl;
    //std::cout << "c111 " << c111 << std::endl;
    //std::cout << "c110 " << c110 << std::endl;

    float x_d = (currentValue[0] - low[0]) / (high[0] - low[0]);
    float y_d = (currentValue[1] - low[1]) / (high[1] - low[1]);
    float z_d = (currentValue[2] - low[2]) / (high[2] - low[2]);

    x_d = x_d > 0 ? x_d : 0;
    y_d = y_d > 0 ? y_d : 0;
    z_d = z_d > 0 ? z_d : 0;

    //std::cout << "x_d " << x_d << std::endl;
    //std::cout << "y_d " << y_d << std::endl;
    //std::cout << "z_d " << z_d << std::endl;

    cv::Vec3f c00 = c000 * (1 - x_d) + c100 * x_d;
    cv::Vec3f c01 = c001 * (1 - x_d) + c101 * x_d;
    cv::Vec3f c11 = c010 * (1 - x_d) + c110 * x_d;
    cv::Vec3f c10 = c011 * (1 - x_d) + c111 * x_d;

    //std::cout << "c00 " << c00 << std::endl;
    //std::cout << "c01 " << c01 << std::endl;
    //std::cout << "c11 " << c11 << std::endl;
    //std::cout << "c10 " << c10 << std::endl;

    cv::Vec3f c0 = c00 * (1 - y_d) + c10 * y_d;
    cv::Vec3f c1 = c01 * (1 - y_d) + c11 * y_d;

    /*std::cout << "c0 " << c0 << std::endl;
    std::cout << "c1 " << c1 << std::endl;*/

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

            //std::cout << "currentValue " << currentValue << std::endl;

            finalValue = interpolation1D(currentValue, low, high);

            //std::cout << "finalValue " << finalValue << std::endl;

            dst.at<cv::Vec3b>(i, j) = lut[finalValue[2]].at<cv::Vec3b>(finalValue[0], finalValue[1]);
        }
    }
}
