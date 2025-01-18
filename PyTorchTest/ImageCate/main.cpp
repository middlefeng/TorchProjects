//
//  main.cpp
//  Image Categorization 
//
//  Created by Dong Feng on 1/10/2025
//

#include <iostream>
#include <torch/torch.h>

#include <memory>

#include "ImageSet.h"
#include "ImageCategoryNet.h"


class SequentialTanhImpl : public torch::nn::Module
{
    
public:
    
    SequentialTanhImpl();
    
    torch::Tensor forward(torch::Tensor x)
    {
        return _sequential->forward(x);
    } //
    
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




const static char* kDataPath = "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin";



void printTensorShape(const torch::Tensor& tensor) {
    std::cout << "Shape: [";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        std::cout << tensor.sizes()[i];
        if (i < tensor.sizes().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}



int main(int argc, const char * argv[])
{
    std::vector<ImageData> data = parseCIFAR10Binary(kDataPath);
    printf("Data Set: %ld.\n", data.size());

    for (size_t i = 0; i < 4; ++i)
    {
        char path[20];
        snprintf(path, 20, "./test%02ld.ppm", i);

        saveAsPPM(path, data[i].data);

        printf("Save Image %02ld, Label %u.\n", i, data[i].label);

    //saveAsPPM("./test0.ppm", data[0].data);
    //saveAsPPM("./test1.ppm", data[1].data);
    //saveAsPPM("./test2.ppm", data[2].data);
    }


    //printf("CUDA Available: %s.\n", torch::cuda::is_available() ? "YES" : "NO");
    //printf("Config: %s.\n", torch::show_config().c_str());

    auto device = torch::device(torch::kCUDA);


    torch::Tensor image1Tensor1 = imageDataToTensor(data[1]);
    torch::Tensor image1Tensor2 = imageDataToTensor(data[2]);
    torch::Tensor imageTensor = torch::stack({image1Tensor1, image1Tensor2}, 0);

    ImageCategoryNet categoryNet;
    categoryNet->to(torch::kCUDA);

    printf("Model created.\n");

    auto categoryTensor = categoryNet(imageTensor.to(torch::kCUDA));

    printTensorShape(categoryTensor);
    printf("Forward pass.\n");


/*
    ImageDataSet dataSet(data);
    //auto batchedDataset = torch::data::datasets::BatchDataset<ImageDataSet, torch::data::Example<>>(
    //    dataSet, 64);

    //auto dataLoader = torch::data::make_data_loader(dataSet
    //);

    auto sampler = torch::data::samplers::RandomSampler(dataSet.size().value());
    // auto sequentialSampler = torch::data::samplers::SequentialSampler(dataSet.size().value());
    auto dataLoader = torch::data::make_data_loader(
        dataSet, sampler, torch::data::DataLoaderOptions().batch_size(64));

    //printf("Number of batches: %ld.\n", dataLoader->size());

    for (std::vector<torch::data::Example<>>& batch : *dataLoader)
    {
        printf("Batch size: %ld.\n", batch.size());
        for (const auto& batchItem : batch) {
            torch::Tensor tensor = batchItem.data;

            printTensorShape(tensor);
        }

        //printTensorShape(batch.label);
    }
    */


    torch::Tensor t_c = torch::tensor({0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0}, device);
    torch::Tensor t_u = torch::tensor({35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4}, device);
    t_c = t_c.unsqueeze(1);
    t_u = t_u.unsqueeze(1);
    
    long long nSamples = t_u.sizes()[0];
    long long nVal = 0.2 * nSamples;
    
    torch::Tensor shuffledIndices = torch::randperm(nSamples, device);
    
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

    tanModel->to(torch::kCUDA);
    lossFunc->to(torch::kCUDA);
    
    training_loop(6001, &optimizer, tanModel, lossFunc,
                  unTraining, unVal, cTraining, cVal);
    
    
    return 0;
}
