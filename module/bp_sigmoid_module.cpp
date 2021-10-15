#include "bp_sigmoid_module.h"

#include "../autograd/linear.h"
#include "../autograd/sigmoid.h"
#include "../autograd/tseloss.h"

BPSigmoidModule::BPSigmoidModule(const std::vector<int>& neurons) {
    for (int i = 0; i < (int)neurons.size() - 1; i++) {
        Linear* fc = new Linear(neurons[i], neurons[i + 1]);
        Sigmoid* s = new Sigmoid();
        sequential.emplace_back(fc);
        sequential.emplace_back(s);
    }
    TSELoss* loss = new TSELoss();
    sequential.emplace_back(loss);
    n_layers = sequential.size();
}
