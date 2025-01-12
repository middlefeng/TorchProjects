

#include "ImageCategoryNet.h"



ImageCategoryNetImpl::ImageCategoryNetImpl()
{
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)));
    pool1 = register_module("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 8, 3).stride(1).padding(1)));
    pool2 = register_module("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

    fc1 = register_module("fc1", torch::nn::Linear(8 * 8 * 8, 32));
    fc2 = register_module("fc2", torch::nn::Linear(32, 10));
}



torch::Tensor ImageCategoryNetImpl::forward(torch::Tensor x)
{
    auto out = conv1(x);
    out = torch::tanh(out);
    out = pool1(out);

    out = conv2(out);
    out = torch::tanh(out);
    out = pool2(out);

    out = out.view({-1, 8 * 8 * 8});
    out = fc1(out);
    out = torch::tanh(out);
    out = fc2(out);

    return out;
}

