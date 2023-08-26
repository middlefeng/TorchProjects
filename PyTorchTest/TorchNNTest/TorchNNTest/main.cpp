//
//  main.cpp
//  TorchNNTest
//
//  Created by Dong Feng on 8/16/23.
//

#include <iostream>
#include <torch/torch.h>

#include <memory>



class SequentialTanhImpl : public torch::nn::Module
{
    
public:
    
    SequentialTanhImpl();
    
    torch::Tensor forward(torch::Tensor x)
    {
        return _sequential->forward(x);
    }
    
private:
    
    torch::nn::Sequential _sequential;
    
};


SequentialTanhImpl::SequentialTanhImpl()
{
    auto sequential = torch::nn::Sequential(torch::nn::Linear(1, 8),
                                            torch::nn::Functional(torch::tanh),
                                            torch::nn::Linear(8, 1));
    
    _sequential = register_module("sequential", sequential);
}


TORCH_MODULE(SequentialTanh);


template <class M, class L>
void training_loop(int epochs, torch::optim::Optimizer* optimizer,
                   torch::nn::ModuleHolder<M>& model,
                   torch::nn::ModuleHolder<L>& lossFunc,
                   torch::Tensor& uTrain, torch::Tensor& uVal,
                   torch::Tensor& cTrain, torch::Tensor& cVal)
{
    for (int e = 0; e < epochs; ++e)
    {
        torch::Tensor pTrain = model(uTrain);
        torch::Tensor lossTrain = lossFunc(pTrain, cTrain);
        
        torch::Tensor pVal = model(uVal);
        torch::Tensor lossVal = lossFunc(pVal, cVal);
        
        optimizer->zero_grad();
        lossTrain.backward();
        optimizer->step();
        
        if (e % 1000 == 0)
        {
            printf("Epoch (%02d): Training Loss: %.2f. Validation Loss: %.2f.\n", e,
                   lossTrain.item<float>(), lossVal.item<float>());
        }
    }
}


int main(int argc, const char * argv[])
{
    torch::Tensor t_c = torch::tensor({0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0});
    torch::Tensor t_u = torch::tensor({35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4});
    t_c = t_c.unsqueeze(1);
    t_u = t_u.unsqueeze(1);
    
    long long nSamples = t_u.sizes()[0];
    long long nVal = 0.2 * nSamples;
    
    torch::Tensor shuffledIndices = torch::randperm(nSamples);
    
    torch::Tensor train_indices = shuffledIndices.slice(0, 0, nSamples - nVal);
    torch::Tensor val_indices = shuffledIndices.slice(0, nSamples - nVal, nSamples);
    
    torch::Tensor uTraining = t_u.index_select(0, train_indices);
    torch::Tensor cTraining = t_c.index_select(0, train_indices);
    
    torch::Tensor uVal = t_u.index_select(0, val_indices);
    torch::Tensor cVal = t_c.index_select(0, val_indices);
    
    torch::Tensor unTraining = 0.1 * uTraining;
    torch::Tensor unVal = 0.1 * uVal;

    
    // model, optimizer, training loop
    
    SequentialTanh tanModel;
    torch::optim::SGD optimizer(tanModel->parameters(), 1e-3);
    
    auto lossFunc = torch::nn::MSELoss();
    
    training_loop(6001, &optimizer, tanModel, lossFunc,
                  unTraining, unVal, cTraining, cVal);
    
    
    return 0;
}
