#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void colorManagementCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& lut, int dimX, int dimY);

int main(int argc, char** argv)
{
    cv::VideoCapture cap("C:\\Users\\ixrplatform\\Documents\\Iacopo\\Complex_Rendering\\video.mp4");

    // Check if video opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create lut
    cv::Mat_<cv::Vec3b> h_lut = cv::imread(argv[1]);
    cv::cuda::GpuMat d_lut;

    const int digit = 64;

    cv::Mat_<cv::Vec3b> lut[256];

    d_lut.upload(h_lut);

    while (1) {

        cv::Mat_<cv::Vec3b> frame;
        cv::Mat_<cv::Vec3b> h_result;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        cv::cuda::GpuMat d_img;
        cv::cuda::GpuMat d_result;

        d_img.upload(frame);
        d_result.upload(frame);

        colorManagementCUDA(d_img, d_result, d_lut, 32, 32);

        d_result.download(h_result);

        // Display the original frame
        cv::imshow("Original video", frame);

        // Display the processed frame
        cv::imshow("Processed video", h_result);

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