#include "bprelunn.h"

#include "../module/bp_relu_module.h"

BPReLUNN::BPReLUNN(const std::vector<int>& neurons) {
    network = new BPReLUModule(neurons);
}
