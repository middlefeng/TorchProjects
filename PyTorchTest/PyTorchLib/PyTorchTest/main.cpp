//
//  main.cpp
//  PyTorchTest
//
//  Created by Dong Feng on 8/6/23.
//

#include <torch/torch.h>
#include <iostream>
#include <vector>



class SelfAttention : public torch::nn::Module
{
    
    torch::nn::Linear tokeys{nullptr}, toqueries{nullptr}, tovalues{nullptr};
    torch::nn::Linear unifyHeads{nullptr};
    
public:
    SelfAttention(int64_t k, int64_t heads = 4)
    {
        
        model->named_children()["name1"];
        
        tokeys = register_module("tokeys", torch::nn::Linear(k, k));
        tokeys->options.bias(false);
        
        toqueries = register_module("toqueries", torch::nn::Linear(k, k));
        toqueries->options.bias(false);
        
        tovalues = register_module("tovalues", torch::nn::Linear(k, k));
        tovalues->options.bias(false);
        
        unifyHeads = register_module("unifyHeads", torch::nn::Linear(k, k));
    }
    
};




torch::Tensor loss_fn(const torch::Tensor& t_p, const torch::Tensor& t_c)
{
    torch::Tensor squared_diffs = torch::pow(t_p - t_c, 2);
    return torch::mean(squared_diffs);
}



torch::Tensor model(const torch::Tensor& t_u, const torch::Tensor& w, const torch::Tensor& b)
{
    return w * t_u + b;
}


/*
torch::Tensor model(const torch::Tensor& t_u, const torch::Tensor& params)
{
    torch::Tensor ones_row = torch::ones({1, t_u.size(0)});
    torch::Tensor t_u_ = torch::cat({t_u, ones_row}, 0);

    return params * t_u_;
}*/



torch::Tensor training(int numEpocs, float learningRate, torch::Tensor& params,
                       torch::Tensor& t_u, torch::Tensor& t_c)
{
    for (int i = 0; i < numEpocs; ++i)
    {
        std::cout << params << std::endl;
        
        if (params.grad().defined())
            params.grad().zero_();
        
        auto loss = loss_fn(model(t_u, params[0], params[1]), t_c);
        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            params -= learningRate * params.grad();
        }
        
        printf("Epoc Loss (%d, %f).\n", i, loss.item<float>());
    }
    
    return params;
}



int main(int argc, const char * argv[])
{
    torch::Tensor t_c = torch::tensor({0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0});
    torch::Tensor t_u = torch::tensor({35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4});
    
    torch::Tensor t_un = 0.1 * t_u;
    
    torch::Tensor params = torch::tensor({1.0, 0.0}, torch::requires_grad(true));
    
    /*
    torch::Tensor loss = loss_fn(model(t_u, params[0], params[1]), t_c);
    loss.backward();
    
    std::cout << params.grad() << std::endl;
     */
    
    params = training(5000, 0.01, params, t_un, t_c);
    std::cout << params << std::endl;
    
    /*torch::Tensor zeros = torch::zeros({3, 4});
    torch::Tensor zeros_mps = zeros.to(torch::kMPS);
    torch::Device device(torch::kCUDA);
    
    std::cout << zeros_mps << std::endl;
    
    printf("Available: %s.\n", torch::hasMPS() ? "YES" : "NO");
    
    std::isnan(1);
    
    std::cout << "Hello, World!\n"; */
    return 0;
}
