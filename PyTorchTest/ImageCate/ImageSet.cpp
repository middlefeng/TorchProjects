

#include <fstream>
#include "ImageSet.h"



std::vector<ImageData> parseCIFAR10Binary(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    std::vector<ImageData> images;

    while (file.peek() != EOF) {
        ImageData img;
        img.data.resize(kImageByte);

        // Read label (1 byte)
        file.read(reinterpret_cast<char*>(&img.label), 1);

        // Read image data (3072 bytes)
        file.read(reinterpret_cast<char*>(img.data.data()), kImageByte);

        // Check for incomplete record
        if (file.gcount() < kImageByte) {
            break;
        }

        images.push_back(std::move(img));
    }

    file.close();
    return images;
}