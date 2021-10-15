#include "sequential_module.h"

#include <cmath>
#include <iostream>

SequentialModule::~SequentialModule() {
    for (auto func : sequential) {
        delete func;
    }
}

void SequentialModule::set_target(const std::vector<std::vector<double>>& target) {
    LossFunction* pf = dynamic_cast<LossFunction*>(sequential[n_layers - 1]);
    if (pf) {
        pf->set_target(target);
    }
}

void SequentialModule::forward(const std::vector<std::vector<double>>& input) {
    for (int i = 0; i < n_layers; i++) {
        if (i) {
            const auto& output = sequential[i - 1]->get_output();
            sequential[i]->forward(output);
        } else {
            sequential[i]->forward(input);
        }
    }
}

void SequentialModule::backward(const std::vector<std::vector<double>>& nothing) {
    for (int i = n_layers - 1; i >= 0; i--) {
        if (i < n_layers - 1) {
            const auto& grad_output = sequential[i + 1]->get_grad_input();
            sequential[i]->backward(grad_output);
        } else {
            sequential[i]->backward(nothing);
        }
    }
}

double SequentialModule::grad_norm() {
    double ret = 0;
    for (AutoGradFunction* func : sequential) {
        ParametrizedFunction* pf = dynamic_cast<ParametrizedFunction*>(func);
        if (pf) {
            double norm = pf->grad_norm();
            ret += norm * norm;
        }
    }
    return sqrt(ret);
}

double SequentialModule::get_loss() {
    return sequential[n_layers - 1]->get_output()[0][0];
}

void SequentialModule::step(double lr) {
    for (AutoGradFunction* func : sequential) {
        ParametrizedFunction* pf = dynamic_cast<ParametrizedFunction*>(func);
        if (pf) {
            pf->step(lr);
        }
    }
}

void SequentialModule::show_params() {
    int i = 0;
    for (AutoGradFunction* func : sequential) {
        ParametrizedFunction* pf = dynamic_cast<ParametrizedFunction*>(func);
        if (pf) {
            std::cout << "params of layer No." << ++i << ":" << std::endl;
            pf->show_params();
            std::cout << std::endl;
        }
    }
}
