#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <string.h>

using namespace std;

class ImageProcessor {
    public:
        ImageProcessor(string image_file, int width, int height, string raw_format, string color_filter_arr, string save_loc);
        ~ImageProcessor(void);

        void DemosaicImage(void);

        // debug settings
        int en_debug;
        int en_timing;

    private:
        FILE *img_fp;
        string save_loc;

        // image meta information
        int width;
        int height;
        int raw_width;
        int raw_img_size;
        uint8_t *raw_img;

        // raw meta
        string raw_format;
        string color_filter_arr;

        void Mipi10ExtractChannels(float *r, float *g, float *b);
        void BilinearInterpolation(float *r, float *g, float *b);
        void BilinearInterpolationCuda(float *r, float *g, float *b);

        // backend proc information
        enum ImageProcBackend {
            cpu = 0,
            cuda = 1
        };

        ImageProcBackend backend; 
        inline void backend_selection(){
            backend = PROC_BACKEND;
            printf("Backend is set to %d\n", backend);
        }
};

#endif // IMAGE_PROCESSING_H
