#include <iostream>
#include <vector>

#include "module.cpp"
using namespace std;

class Model {
public:
    virtual void train(const vector<vector<double>>&, const vector<vector<double>>&, double, double) = 0;
    virtual void test(const vector<vector<double>>&, const vector<vector<double>>&) = 0;
    virtual void show_params() = 0;
    virtual bool stop(double, double, double) = 0;
    virtual ~Model() {
    }
};

class SequentialModel : public Model {
protected:
    SequentialModule* network;

public:
    void train(const vector<vector<double>>& train_input, const vector<vector<double>>& train_target, double eps,
               double lr) {
        int iter = 0;
        double loss, grad;
        while (1) {
            network->set_target(train_target);
            network->forward(train_input);
            loss = network->get_loss();
            network->backward(vector<vector<double>>(0));
            grad = network->grad_norm();
            if (iter % 1000 == 0) {
                cout << "iter = " << iter << " loss = " << loss << " grad = " << grad << endl;
            }
            if (stop(loss, grad, eps))
                break;
            network->step(lr);
            iter++;
        }
    }

    void test(const vector<vector<double>>& test_input, const vector<vector<double>>& test_target) {
        network->set_target(test_target);
        network->forward(test_input);
        double loss = network->get_loss();
        cout << "test: tseloss = " << loss << endl;
    }

    void show_params() {
        network->show_params();
    }

    bool stop(double loss, double grad, double eps) {
        return loss < eps || grad < eps;
    }

    ~SequentialModel() {
        delete network;
    }
};

class BPNN : public SequentialModel {
public:
    BPNN(const vector<int>& neurons) {
        network = new BPSigmoidModule(neurons);
    }
};

class BPReLUNN : public SequentialModel {
public:
    BPReLUNN(const vector<int>& neurons) {
        network = new BPReLUModule(neurons);
    }
};
