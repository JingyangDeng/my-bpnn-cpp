#ifndef _MODEL_H
#define _MODEL_H

#include <vector>

class Model {
public:
    virtual void train(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&, double,
                       double) = 0;
    virtual void test(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&) = 0;
    virtual void show_params() = 0;
    virtual bool stop(double, double, double) = 0;
    virtual ~Model();
};

#endif
