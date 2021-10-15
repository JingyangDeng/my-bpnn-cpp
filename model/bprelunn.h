#ifndef _BPRELUNN
#define _BPRELUNN

#include <vector>

#include "sequential_model.h"

class BPReLUNN : public SequentialModel {
public:
    BPReLUNN(const std::vector<int>& neurons);
};

#endif
