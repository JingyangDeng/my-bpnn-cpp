#include "bp_relu_module.h"

#include "../autograd/linear.h"
#include "../autograd/relu.h"
#include "../autograd/tseloss.h"

BPReLUModule::BPReLUModule(const std::vector<int>& neurons) {
    for (int i = 0; i < (int)neurons.size() - 1; i++) {
        Linear* fc = new Linear(neurons[i], neurons[i + 1]);
        ReLU* r = new ReLU();
        sequential.emplace_back(fc);
        sequential.emplace_back(r);
    }
    TSELoss* loss = new TSELoss();
    sequential.emplace_back(loss);
    n_layers = sequential.size();
}
