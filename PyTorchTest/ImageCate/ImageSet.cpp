

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



void saveAsPPM(const std::string& fileName, const std::vector<uint8_t>& imageData)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if (!outFile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    // Write the PPM header
    outFile << "P6\n";
    outFile << "32 32\n";
    outFile << "255\n";

    for (size_t index = 0; index < kImageSize * kImageSize; ++index)
    {
        uint8_t pixel[3];
        pixel[0] = imageData[index];
        pixel[1] = imageData[index + kImageSize * kImageSize];
        pixel[2] = imageData[index + kImageSize * kImageSize * 2];

        outFile.write(reinterpret_cast<const char*>(pixel), 3);
    }

    outFile.close();
    std::cout << "Image saved to " << fileName << "\n";
}

