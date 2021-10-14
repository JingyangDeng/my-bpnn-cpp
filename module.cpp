#include <vector>

#include "autograd.cpp"
using namespace std;

class Module : public ParametrizedFunction, public LossFunction, public AutoGradFunction {
public:
    virtual ~Module() {
    }
};

class SequentialModule : public Module {
protected:
    int n_layers;
    vector<BasicFunction*> sequential;

public:
    ~SequentialModule() {
        for (auto func : sequential) {
            delete func;
        }
    }

    void set_target(const vector<vector<double>>& target) {
        LossFunction* pf = dynamic_cast<LossFunction*>(sequential[n_layers - 1]);
        if (pf) {
            pf->set_target(target);
        }
    }

    void forward(const vector<vector<double>>& input) {
        for (int i = 0; i < n_layers; i++) {
            if (i) {
                const auto& output = sequential[i - 1]->get_output();
                sequential[i]->forward(output);
            } else {
                sequential[i]->forward(input);
            }
        }
    }

    void backward(const vector<vector<double>>& nothing) {
        for (int i = n_layers - 1; i >= 0; i--) {
            if (i < n_layers - 1) {
                const auto& grad_output = sequential[i + 1]->get_grad_input();
                sequential[i]->backward(grad_output);
            } else {
                sequential[i]->backward(nothing);
            }
        }
    }

    double grad_norm() {
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

    double get_loss() {
        return sequential[n_layers - 1]->get_output()[0][0];
    }

    void step(double lr) {
        for (AutoGradFunction* func : sequential) {
            ParametrizedFunction* pf = dynamic_cast<ParametrizedFunction*>(func);
            if (pf) {
                pf->step(lr);
            }
        }
    }

    void show_params() {
        int i = 0;
        for (AutoGradFunction* func : sequential) {
            ParametrizedFunction* pf = dynamic_cast<ParametrizedFunction*>(func);
            if (pf) {
                cout << "params of layer No." << ++i << ":" << endl;
                pf->show_params();
                cout << endl;
            }
        }
    }
};

class BPSigmoidModule : public SequentialModule {
public:
    BPSigmoidModule(const vector<int>& neurons) {
        for (int i = 0; i < (int)neurons.size() - 1; i++) {
            Linear* fc = new Linear(neurons[i], neurons[i + 1]);
            Sigmoid* s = new Sigmoid();
            sequential.emplace_back(fc);
            sequential.emplace_back(s);
        }
        TSELoss* loss = new TSELoss();
        sequential.emplace_back(loss);
        n_layers = sequential.size();
    }
};

class BPReLUModule : public SequentialModule {
public:
    BPReLUModule(const vector<int>& neurons) {
        for (int i = 0; i < (int)neurons.size() - 1; i++) {
            Linear* fc = new Linear(neurons[i], neurons[i + 1]);
            ReLU* r = new ReLU();
            sequential.emplace_back(fc);
            sequential.emplace_back(r);
        }
        TSELoss* loss = new TSELoss();
        sequential.emplace_back(loss);
        n_layers = sequential.size();
    }
};
