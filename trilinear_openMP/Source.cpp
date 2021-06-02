#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void colorManagement(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b> lut[]);

int main(int argc, char** argv)
{
    cv::VideoCapture cap("C:\\Users\\cl11273v\\Desktop\\video.mp4");

    // Check if video opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create lut
    cv::Mat_<cv::Vec3b> dst_lut = cv::imread(argv[1]);

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
            }

    while (1) {

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        cv::Mat_<cv::Vec3b> src = frame;
        cv::Mat_<cv::Vec3b> dst;

        colorManagement(src, dst, lut);

        // Display the original frame
        cv::imshow("Original video", frame);

        // Display the processed frame
        cv::imshow("Processed video", dst);

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}