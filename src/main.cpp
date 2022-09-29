#include <stdio.h>
#include <getopt.h>
#include <iostream>

// Project includes
#include "config_file.h"
#include "image_processing.h"

static int en_debug = 0;
static int en_timing = 0;


// -----------------------------------------------------------------------------------------------------------------------------
// Print the help message
// -----------------------------------------------------------------------------------------------------------------------------
static void _print_usage() {
    printf("Usage: ./run.sh <options>\n");
    printf("Options:\n");
    printf("-d      Show extra debug messages\n");
    printf("-t      Enable debug timing messages\n");
    printf("-h      Show this help menu\n");
}


// -----------------------------------------------------------------------------------------------------------------------------
// Parse args
// -----------------------------------------------------------------------------------------------------------------------------
static int _parse_opts(int argc, char* argv[]) {
	static struct option long_options[] =
	{
		{"debug",				no_argument,	0,	'd'},
		{"timing",				no_argument,	0,	't'},
		{"help",				no_argument,	0,	'h'},
		{0, 0, 0, 0}
	};

	while(1){
		int option_index = 0;
		int c = getopt_long(argc, argv, "dth", long_options, &option_index);

		if(c == -1) break; // Detect the end of the options.

		switch(c){
		case 0:
			// for long args without short equivalent that just set a flag
			// nothing left to do so just break.
			if (long_options[option_index].flag != 0) break;
			break;

		case 'd':
			en_debug = 1;
            printf("[INFO] Enabling debug mode\n");
			break;

		case 't':
			en_timing = 1;
            printf("[INFO] Enabling debug timing\n");
			break;

		case 'h':
			_print_usage();
			return -1;

		default:
			_print_usage();
			return -1;
		}
	}

	return 0;
}


int main(int argc, char* argv[]) {

	// check for options
	if(_parse_opts(argc, argv)) return -1;

    // Initialize image processor
    ImageProcessor *img_proc = new ImageProcessor(image_file, width, height, format, cfa, save_loc);
    img_proc->en_debug = en_debug;
    img_proc->en_timing = en_timing;

    // Begin image demosaicing
    img_proc->DemosaicImage();

    // cleanup
    delete img_proc;

    return 0;
}
