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
#include "TorchUtils.h"


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




const static char* kDataPath[] = {
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin",
};

const static char* kDataValidatePath[] = {
    "../../../../train_data/cifar-10-binary/cifar-10-batches-bin/test_batch.bin"
};



std::vector<ImageData> originalData()
{
    std::vector<ImageData> result;

    for (const char* dataPath : kDataPath)
    {
        std::vector<ImageData> oneSet = parseCIFAR10Binary(dataPath);
        result.insert(result.end(), oneSet.begin(), oneSet.end());
    }

    return result;
}



std::vector<ImageData> validateData()
{
    std::vector<ImageData> result;

    for (const char* dataPath : kDataValidatePath)
    {
        std::vector<ImageData> oneSet = parseCIFAR10Binary(dataPath);
        result.insert(result.end(), oneSet.begin(), oneSet.end());
    }

    return result;
}



/*
void printTensorShape(const torch::Tensor& tensor) {
    std::cout << "Shape: [";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        std::cout << tensor.sizes()[i];
        if (i < tensor.sizes().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}*/



int main(int argc, const char * argv[])
{
    std::vector<ImageData> data = originalData(); // parseCIFAR10Binary(kDataPath[0]);
    printf("Data Set: %ld.\n", data.size());

    std::vector<ImageData> validateDataVector = validateData();

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


    /*torch::Tensor image1Tensor1 = imageDataToTensor(data[1]);
    torch::Tensor image1Tensor2 = imageDataToTensor(data[2]);
    torch::Tensor imageTensor = torch::stack({image1Tensor1, image1Tensor2}, 0);*/

    printf("Before create model.\n");

    ImageCategoryNet categoryNet;
    categoryNet->to(torch::kCUDA);

    printf("Model created.\n");

    //auto categoryTensor = categoryNet(imageTensor.to(torch::kCUDA));

    //printTensorShape(categoryTensor);
    //printf("Forward pass.\n");



    ImageDataSet dataSet(data);
    ImageDataSet validateDataSet(validateDataVector);
    //auto batchedDataset = torch::data::datasets::BatchDataset<ImageDataSet, torch::data::Example<>>(
    //    dataSet, 64);

    //auto dataLoader = torch::data::make_data_loader(dataSet
    //);

    auto sampler = torch::data::samplers::RandomSampler(dataSet.size().value());
    // auto sequentialSampler = torch::data::samplers::SequentialSampler(dataSet.size().value());
    auto dataLoader = torch::data::make_data_loader(
        dataSet, sampler, torch::data::DataLoaderOptions().batch_size(64));

    //printf("Number of batches: %ld.\n", dataLoader->size());

    std::shared_ptr<torch::optim::SGD> optimizer = std::make_shared<torch::optim::SGD>(categoryNet->parameters(), torch::optim::SGDOptions(0.02).momentum(0.5));

    std::vector<torch::Tensor> validateTensorList;
    std::vector<torch::Tensor> validateLabelList;
    for (size_t index = 0; index < validateDataSet.size(); ++index)
    {
        validateTensorList.push_back(validateDataSet.get(index).data);
        validateLabelList.push_back(validateDataSet.get(index).target);
    }
    auto batchTensor = torch::stack(validateTensorList, 0);
    auto categoryLabels = torch::stack(validateLabelList, 0);

    


    for (int epoch = 0; epoch < 10000; ++epoch)
    {
        float lossValue = 0;
        size_t batchNumber = 0;

        for (std::vector<torch::data::Example<>>& batch : *dataLoader)
        {
            std::vector<torch::Tensor> batchTensorList;
            std::vector<torch::Tensor> labelTensorList;

            //printf("Batch size: %ld.\n", batch.size());
            for (const auto& batchItem : batch) {
                batchTensorList.push_back(batchItem.data);
                labelTensorList.push_back(batchItem.target);
            }

            auto batchTensor = torch::stack(batchTensorList, 0);
            auto categoryResult = categoryNet(batchTensor.to(torch::kCUDA));
            auto categoryLabels = torch::stack(labelTensorList, 0);

            torch::nn::CrossEntropyLoss lossFunction;
            auto loss = lossFunction(categoryResult, categoryLabels.to(torch::kCUDA));

            optimizer->zero_grad();
            loss.backward();

            optimizer->step();

            lossValue += loss.item<float>();
            batchNumber += 1;

            


        }



        lossValue /= (float)batchNumber;
        printf("Epoch %d: Loss %f.\n", epoch, lossValue);

        if (lossValue < 0.5)
        {
            optimizer = std::make_shared<torch::optim::SGD>(categoryNet->parameters(), torch::optim::SGDOptions(0.005).momentum(0.1));
        }

        if (lossValue < 0.3)
        {
            optimizer = std::make_shared<torch::optim::SGD>(categoryNet->parameters(), torch::optim::SGDOptions(0.002));
        }

        if (lossValue < 0.2)
        {
            optimizer = std::make_shared<torch::optim::SGD>(categoryNet->parameters(), torch::optim::SGDOptions(0.0005));
        }


        {
            torch::NoGradGuard no_grad_guard;

            auto categoryResult = categoryNet(batchTensor.to(torch::kCUDA));
        
            torch::nn::CrossEntropyLoss lossFunction;
            auto loss = lossFunction(categoryResult, categoryLabels.to(torch::kCUDA));

            printf("Validate loss: %f.\n", loss.item<float>());
        }


        fflush(stdout);
    
    }
    
    
    return 0;
}
