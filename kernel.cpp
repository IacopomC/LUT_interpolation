#include <opencv2/opencv.hpp>
#include <math.h>

cv::Vec3b interpolation()
{

}


void colorManagement(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b> lut[])
{

    dst.create(src.rows, src.cols);
    dst = cv::Vec3b(0, 0, 0);

    cv::Vec3f currentValue;

    #pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {

            currentValue = src.at<cv::Vec3b>(i, j) / 4;

            std::cout << "currentValue " << currentValue << std::endl;

            //dst.at<cv::Vec3b>(i, j) = lut[currentValue[2]].at<cv::Vec3b>(currentValue[0], currentValue[1]);
        }
    }
}
