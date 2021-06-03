#include <opencv2/opencv.hpp>
#include <math.h>

cv::Vec3b nearest_neighbor(cv::Vec3f currentValue)
{
    cv::Vec3f c;

    cv::Vec3f low = cv::Vec3b(floor(currentValue[0]), floor(currentValue[1]), floor(currentValue[2]));
    cv::Vec3f high = cv::Vec3b(ceil(currentValue[0]), ceil(currentValue[1]), ceil(currentValue[2]));

    float x_d = (currentValue[0] - low[0]) / (high[0] - low[0]);
    float y_d = (currentValue[1] - low[1]) / (high[1] - low[1]);
    float z_d = (currentValue[2] - low[2]) / (high[2] - low[2]);

    x_d = x_d > 0 ? x_d : 0;
    y_d = y_d > 0 ? y_d : 0;
    z_d = z_d > 0 ? z_d : 0;


    c[0] = x_d < 0.5 ? low[0] : high[0];
    c[1] = y_d < 0.5 ? low[1] : high[1];

    c[2] = z_d < 0.5 ? low[2] : high[2];

    return c;
}


void colorManagement(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b> lut[])
{

    dst.create(src.rows, src.cols);
    dst = cv::Vec3b(0, 0, 0);

    cv::Vec3f currentValue;

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

            finalValue = nearest_neighbor(currentValue);

            dst.at<cv::Vec3b>(i, j) = lut[finalValue[2]].at<cv::Vec3b>(finalValue[0], finalValue[1]);
        }
    }
}
