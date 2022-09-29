#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <iostream>


/*
* This file contains configuration that's specific to the raw image pipeline
*
* parameters description:
*
* width     : width of the image
* height    : height of the image
* cfa       : Color filter arrangement
* format    : Currently only support mipi raw (raw10 opaque)
* save_loc  : Location to save the image
* image_file: Path to the image file
*/

static int width = 1280;
static int height = 800;
static std::string cfa = "null"; // not yet supported
static std::string format = "mipi_raw"; // currently only supported format
static std::string save_loc = "images/result.png";
static std::string image_file = "images/test_image.raw";

#endif // CONFIG_FILE_H