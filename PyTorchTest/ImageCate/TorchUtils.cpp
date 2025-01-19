

#include "TorchUtils.h"



std::string tensorShape(torch::Tensor& tensor)
{
    std::string result = "[";
    char dimensionStr[16];
    for (size_t i = 0; i < tensor.sizes().size(); ++i)
    {
        if (i < tensor.sizes().size() - 1)
            snprintf(dimensionStr, 16, "%ld, ", tensor.sizes()[i]);
        else
            snprintf(dimensionStr, 16, "%ld]", tensor.sizes()[i]);

        result += dimensionStr;
    }

    return result;
}



std::string tensorInfomration(torch::Tensor& tensor, int indentation)
{
    std::ostringstream oss;
    std::string indent;
    
    if (indentation > 0)
        indent = std::string(indentation, ' ');

    auto tensor_data = tensor.to(torch::kCPU);
    oss << indent << "[";
    
    // Access tensor data based on type
    if (tensor.dtype() == torch::kFloat32)
    {
        auto accessor = tensor_data.accessor<float, 1>();
        for (int64_t i = 0; i < tensor_data.numel() - 1; ++i)
            oss << accessor[i] << ", ";
        
        oss << accessor[tensor_data.numel() - 1] << "]";
    }

    oss.flush();

    return oss.str();
}

