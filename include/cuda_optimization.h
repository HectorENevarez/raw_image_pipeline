#ifndef CUDA_OPTIMIZATION_H
#define CUDA_OPTIMIZATION_H

#include <opencv2/imgproc/imgproc.hpp>

int convolution_2d_cuda(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b, cv::Mat& demosaiced);

#endif // CUDA_OPTIMIZATION_H