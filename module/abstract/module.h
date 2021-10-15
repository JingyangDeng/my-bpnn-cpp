#ifndef _MODULE_H
#define _MODULE_H

#include "../../autograd/abstract/autograd_function.h"
#include "../../autograd/abstract/loss_function.h"
#include "../../autograd/abstract/parametrized_function.h"

class Module : public ParametrizedFunction, public LossFunction, public AutoGradFunction {
public:
    virtual ~Module();
};

#endif
