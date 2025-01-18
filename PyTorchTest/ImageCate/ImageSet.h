


#ifndef __IMAGE_SET_H_
#define __IMAGE_SET_H_


#include <vector>
#include <torch/torch.h>


const static size_t kImageByte = 32 * 32 * 3;


struct ImageData
{
    uint8_t label;              // The label of the image
    std::vector<uint8_t> data;  // The image data (RGB flattened)
};


class ImageDataSet : public torch::data::datasets::Dataset<ImageDataSet>
{
private:
    std::vector<ImageData> _imageData;

    //std::vector<size_t> _indices;
    //std::mt19937 _rng; // Random number generator

public:
    explicit ImageDataSet(const std::vector<ImageData>& data);
    
    torch::optional<size_t> size() const override;

    torch::data::Example<> get(size_t index) override;
    //std::vector<torch::data::Example<>> get_batch(torch::ArrayRef<size_t> indices) override;

    // void reset() override;
};


extern std::vector<ImageData> parseCIFAR10Binary(const std::string& filePath);
extern void saveAsPPM(const std::string& fileName, const std::vector<uint8_t>& imageData);
extern torch::Tensor imageDataToTensor(const ImageData& data);


#endif
