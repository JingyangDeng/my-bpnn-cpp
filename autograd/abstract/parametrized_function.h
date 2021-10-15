#ifndef _PARAMETRIZED_FUNCTION_H
#define _PARAMETRIZED_FUNCTION_H

class ParametrizedFunction {
public:
    virtual void step(double) = 0;
    virtual double grad_norm() = 0;
    virtual void show_params() = 0;
    virtual ~ParametrizedFunction();
};

#endif
