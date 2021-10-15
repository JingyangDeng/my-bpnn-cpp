#include "bpnn.h"

#include "../module/bp_sigmoid_module.h"

BPNN::BPNN(const std::vector<int>& neurons) {
    network = new BPSigmoidModule(neurons);
}
