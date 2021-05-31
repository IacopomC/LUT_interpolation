#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void colorManagement(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b> lut[]);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> src = cv::imread(argv[1]);
    cv::Mat_<cv::Vec3b> dst;
    cv::Mat_<cv::Vec3b> dst_lut = cv::imread(argv[2]);

    const int digit = 64;

    cv::Mat_<cv::Vec3b> lut[256];

    for (int i = 0; i < 256; i++) {
        lut[i] = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
    }

    float step = 1.0 / (digit - 1.0);
    for (int r = 0; r < digit; r++)
        for (int g = 0; g < digit; g++)
           for (int b = 0; b < digit; b++)
            {
                lut[r].at<cv::Vec3b>(b, g) = dst_lut(r, g * digit + b);
                //std::cout << "lut " << lut[r].at<cv::Vec3b>(b, g) << std::endl;
            }

    colorManagement(src, dst, lut);
    
    //std::cout << "lut " << lut << std::endl;


    cv::imshow("Original Image", src);
    cv::imshow("Processed Image", dst);

    cv::waitKey();
    return 0;
}