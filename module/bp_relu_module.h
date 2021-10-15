#ifndef _BP_RELU_MODULE_H
#define _BP_RELU_MODULE_H

#include <vector>

#include "sequential_module.h"

class BPReLUModule : public SequentialModule {
public:
    BPReLUModule(const std::vector<int>& neurons);
};

#endif
