

#ifndef __IMAGE_CATEGORY_H__
#define __IMAGE_CATEGORY_H__


#include <torch/torch.h>


class ImageCategoryNetImpl : public torch::nn::Module
{
    
public:
    
    ImageCategoryNetImpl();
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::MaxPool2d pool1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::MaxPool2d pool2 = nullptr;
    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;
    
};

TORCH_MODULE(ImageCategoryNet);


#endif