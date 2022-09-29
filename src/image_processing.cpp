#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <iostream>

// Opencv includes
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Project includes
#include "image_processing.h"


// -----------------------------------------------------------------------------------------------------------------------------
//  Obtain current monotonic time
// -----------------------------------------------------------------------------------------------------------------------------
static int64_t _apps_time_monotonic_ns(){
	struct timespec ts;
	if(clock_gettime(CLOCK_MONOTONIC, &ts)){
		fprintf(stderr,"ERROR calling clock_gettime\n");
		return -1;
	}
	return (int64_t)ts.tv_sec*1000000000 + (int64_t)ts.tv_nsec;
}


// -----------------------------------------------------------------------------------------------------------------------------
//  Manages debug timing start, stop, and message printing
//  action - 1 : start timinig
//  action - 0 : end timing and print debug message
// -----------------------------------------------------------------------------------------------------------------------------
enum timingAction {
    stop_timer = 0,
    start_timer
};

static void _debug_timing_manager(timingAction action){
    static int64_t segment_start = 0;
    if (action == start_timer){
        segment_start = _apps_time_monotonic_ns();
    }
    else if (action == stop_timer){
        int64_t segment_process = _apps_time_monotonic_ns() - segment_start;
        printf("Segment process time: %6.2fms\n", ((double)segment_process)/1000000.0);
    }
    else {
        printf("[ERROR] timing action is not supported: %d\n", action);
    }
}


// -----------------------------------------------------------------------------------------------------------------------------
//  mipi10 opaque raw is stored in the format of:
//  P3(1:0) P2(1:0) P1(1:0) P0(1:0) P3(9:2) P2(9:2) P1(9:2) P0(9:2)
//  4 pixels occupy 5 bytes, no padding needed
// -----------------------------------------------------------------------------------------------------------------------------
void ImageProcessor::Mipi10ExtractChannels(float *r, float *g, float *b){
    int seg_a, seg_b, seg_c, seg_d, seg_e;


    if (en_timing) _debug_timing_manager(start_timer);
    for (int idx = 0; idx < (raw_img_size/5); idx++){
        // Unpack mipi raw10 into raw8
        seg_a = raw_img[idx*5];
        seg_b = raw_img[(idx*5)+1];
        seg_c = raw_img[(idx*5)+2];
        seg_d = raw_img[(idx*5)+3];
        seg_e = raw_img[(idx*5)+4];

        // Extract color channels for image
        // This step is different for each CFA but a BGGR pixel arrangement would look as following
        // b g b g ...
        // g r g r ...
        // b g b g ...
        if (((idx*4) / width) % 2) {
            g[idx*4]     = ((seg_a << 2) + ((seg_e >> 0) & 0x03)) / 1024.0;
            b[(idx*4)+1] = ((seg_b << 2) + ((seg_e >> 0) & 0x03)) * 1.8 / 1024.0;
            g[(idx*4)+2] = ((seg_c << 2) + ((seg_e >> 0) & 0x03)) / 1024.0;
            b[(idx*4)+3] = ((seg_d << 2) + ((seg_e >> 0) & 0x03)) * 1.8 / 1024.0;
        }
        else {
            r[idx*4]     = ((seg_a << 2) + ((seg_e >> 0) & 0x03)) * 1.8 / 1024.0;
            g[(idx*4)+1] = ((seg_b << 2) + ((seg_e >> 0) & 0x03)) / 1024.0;
            r[(idx*4)+2] = ((seg_c << 2) + ((seg_e >> 0) & 0x03)) * 1.8 / 1024.0;
            g[(idx*4)+3] = ((seg_d << 2) + ((seg_e >> 0) & 0x03)) / 1024.0;
        }
    }
    if (en_timing) _debug_timing_manager(stop_timer);
}


// -----------------------------------------------------------------------------------------------------------------------------
// Bilinear interpolation algorithm for debayering raw images
// -----------------------------------------------------------------------------------------------------------------------------
void ImageProcessor::BilinearInterpolation(float *r, float *g, float *b){
    if (en_timing) _debug_timing_manager(start_timer);
    
    // convert to array to mat
    cv::Mat r_mat(800, 1280, CV_32F, r);
    cv::Mat g_mat(800, 1280, CV_32F, g);
    cv::Mat b_mat(800, 1280, CV_32F, b);

    // define kernels
    cv::Mat H_G =( cv::Mat_<float>(3,3) << 0, 1, 0, 1, 4, 1, 0, 1, 0); 
    H_G *= (1.0/4.0);

    cv::Mat H_RB = ( cv::Mat_<float>(3,3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    H_RB *= (1.0/4.0);

    std::vector<cv::Mat> rgb(3);
    rgb[0].push_back(r_mat);
    rgb[1].push_back(g_mat);
    rgb[2].push_back(b_mat);

    cv::filter2D(rgb[0], rgb[0], CV_32F, H_RB);
    cv::filter2D(rgb[1], rgb[1], CV_32F, H_G);
    cv::filter2D(rgb[2], rgb[2], CV_32F, H_RB);

    cv::Mat demosaiced;
    cv::merge(rgb, demosaiced);
    demosaiced.convertTo(demosaiced, CV_8UC3, 255.0);

    if (en_timing) _debug_timing_manager(stop_timer);

    // Write image to output
    cv::imwrite(save_loc, demosaiced);
}


// -----------------------------------------------------------------------------------------------------------------------------
// Begin demosaicing image
// -----------------------------------------------------------------------------------------------------------------------------
void ImageProcessor::DemosaicImage() {

    float *r = new float[raw_img_size];
    float *b = new float[raw_img_size];
    float *g = new float[raw_img_size];

    Mipi10ExtractChannels(r, g, b);

    BilinearInterpolation(r, g, b);

    // cleanup
    delete[] r;
    delete[] g;
    delete[] b;
}


// -----------------------------------------------------------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------------------------------------------------------
ImageProcessor::ImageProcessor(string image_file, int width, int height, string raw_format, string color_filter_arr, string save_loc) {
    int ret;
    
    img_fp = fopen(image_file.c_str() , "rb");
    if (img_fp == NULL) {
        fprintf(stderr, "Failed to open image: %s\n", strerror(errno));
        // TODO: handle errors
    }

    this->width = width;
    this->height = height;
    this->color_filter_arr;
    this->raw_format = raw_format;
    this->save_loc = save_loc;

    // TODO: need support for other raw formats
    if (raw_format == "mipi_raw") raw_width = width + (width / 4);

    raw_img_size = raw_width * height;

    // Allocate memory for image
    raw_img = new uint8_t[raw_img_size];

    ret = fread(raw_img, 1, raw_img_size, img_fp);
    if (ret != raw_width*height){
        fprintf(stderr, "Failed to read complete raw image: %s\n", strerror(errno));
        // TODO: handle errors
    }
}


// -----------------------------------------------------------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------------------------------------------------------
ImageProcessor::~ImageProcessor() {
    fclose(img_fp);
    delete [] raw_img;
}

