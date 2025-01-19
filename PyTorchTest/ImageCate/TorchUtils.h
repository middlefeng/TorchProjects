

#ifndef __TORCH_UTILS_H__
#define __TORCH_UTILS_H__


#include <string>
#include <torch/torch.h>


std::string tensorShape(torch::Tensor& tensor);
std::string tensorInfomration(torch::Tensor& tensor, int indentation);


#endif

