


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

public:
    explicit ImageDataSet(const std::vector<ImageData>& data);
    
    torch::optional<size_t> size() const override;

    torch::data::Example<> get(size_t index) override;
};


extern std::vector<ImageData> parseCIFAR10Binary(const std::string& filePath);
extern void saveAsPPM(const std::string& fileName, const std::vector<uint8_t>& imageData);
extern torch::Tensor imageDataToTensor(const ImageData& data);


#endif
