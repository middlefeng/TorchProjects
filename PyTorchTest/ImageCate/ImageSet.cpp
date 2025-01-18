

#include <fstream>
#include <iostream>
#include "ImageSet.h"

#include <stdio.h>


static const int kImageSize = 32;   // Image dimension (32x32)
static const int kChannels = 3;     // RGB channels
static const int kImageBytes = kImageSize * kImageSize * kChannels; // Total bytes per image


std::vector<ImageData> parseCIFAR10Binary(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    std::vector<ImageData> images;

    while (file.peek() != EOF)
    {
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



torch::Tensor imageDataToTensor(const ImageData& data)
{
    auto dataPtr = const_cast<uint8_t*>(data.data.data());
    auto tensor = torch::from_blob(reinterpret_cast<void*>(dataPtr), {3, 32, 32}, torch::kUInt8);
    auto tensorFloat = tensor.to(torch::kFloat32) / 255.f;
    return tensorFloat;
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
}



ImageDataSet::ImageDataSet(const std::vector<ImageData>& data)
: _imageData(data)
{
}


torch::optional<size_t> ImageDataSet::size() const
{
    return _imageData.size();
}


torch::data::Example<> ImageDataSet::get(size_t index)
{
    printf("Get data item: %ld.\n", index);

    auto data = _imageData[index];
    auto dataTensor = imageDataToTensor(data);
    auto labelTensor = torch::tensor(static_cast<int64_t>(data.label), torch::kLong);

    return {dataTensor, labelTensor};
}


/*std::vector<torch::data::Example<>> ImageDataSet::get_batch(torch::ArrayRef<size_t> indices)
 {
    std::vector<torch::data::Example<>> batch;
    for (auto index : indices)
    {
         batch.push_back(get(index));
    }
    return batch;
}*/


/*void ImageDataSet::reset() override
{
    std::shuffle(_indices.begin(), _indices.end(), _rng); // Shuffle indices
}*/

