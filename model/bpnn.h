#ifndef _BPNN_H
#define _BPNN_H

#include <vector>

#include "sequential_model.h"

class BPNN : public SequentialModel {
public:
    BPNN(const std::vector<int>& neurons);
};

#endif
