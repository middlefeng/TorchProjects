


#ifndef __IMAGE_SET_H_
#define __IMAGE_SET_H_


#include <vector>


const static size_t kImageByte = 32 * 32 * 3;


struct ImageData
{
    uint8_t label;              // The label of the image
    std::vector<uint8_t> data;  // The image data (RGB flattened)

};


#endif
