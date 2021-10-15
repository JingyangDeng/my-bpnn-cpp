#ifndef _BP_SIGMOID_MODULE_H
#define _BP_SIGMOID_MODULE_H

#include <vector>

#include "sequential_module.h"

class BPSigmoidModule : public SequentialModule {
public:
    BPSigmoidModule(const std::vector<int>& neurons);
};

#endif
