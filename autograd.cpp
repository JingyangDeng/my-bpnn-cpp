#include <iostream>
#include <vector>

#include "math.cpp"
#define ALPHA 0.1
#define MAX 1000
using namespace std;

class AutoGradFunction {
public:
    virtual void forward(const vector<vector<double>>&) = 0;
    virtual void backward(const vector<vector<double>>&) = 0;
    virtual ~AutoGradFunction() {
    }
};

class BasicFunction : public AutoGradFunction {
protected:
    vector<vector<double>> output;
    vector<vector<double>> grad_input;

public:
    const vector<vector<double>>& get_output() {
        return output;
    }

    const vector<vector<double>>& get_grad_input() {
        return grad_input;
    }

    virtual ~BasicFunction() {
    }
};

class ParametrizedFunction {
public:
    virtual void step(double) = 0;
    virtual double grad_norm() = 0;
    virtual void show_params() = 0;
    virtual ~ParametrizedFunction() {
    }
};

class LossFunction {
public:
    virtual void set_target(const vector<vector<double>>&) = 0;
    virtual double get_loss() = 0;
    virtual ~LossFunction() {
    }
};

class Linear : public ParametrizedFunction, public BasicFunction {
private:
    vector<vector<double>> weight;
    vector<vector<double>> input;        // [n_(k-1), b]
    vector<vector<double>> grad_weight;  // [nk, n_(k-1) + 1]

    void reset_params() {
        for (int i = 0; i < (int)weight.size(); i++) {
            for (int j = 0; j < (int)weight[0].size(); j++) {
                weight[i][j] = ALPHA * ((rand() % MAX) / (double)MAX - 0.5);
            }
        }
    }

public:
    Linear(int in_features, int out_features) {
        weight.resize(out_features, vector<double>(in_features + 1));
        reset_params();
    }

    void forward(const vector<vector<double>>& input) {
        this->input = input;
        linear(weight, input, output);
    }

    double grad_norm() {
        return norm(grad_weight);
    }

    void show_params() {
        for (int i = 0; i < (int)weight.size(); i++) {
            for (int j = 0; j < (int)weight[0].size(); j++) {
                cout << weight[i][j] << " ";
            }
            cout << endl;
        }
    }

    void backward(const vector<vector<double>>& grad_output) {
        linear_back(grad_output, input, weight, grad_input, grad_weight);
    }

    void step(double lr) {
        update(weight, grad_weight, lr);
    }
};

class Sigmoid : public BasicFunction {
public:
    void forward(const vector<vector<double>>& input) {
        sigmoid(input, output);
    }

    void backward(const vector<vector<double>>& grad_output) {
        sigmoid_back(grad_output, output, grad_input);
    }
};

class ReLU : public BasicFunction {
private:
    vector<vector<double>> input;

public:
    void forward(const vector<vector<double>>& input) {
        this->input = input;
        relu(input, output);
    }

    void backward(const vector<vector<double>>& grad_output) {
        relu_back(grad_output, input, grad_input);
    }
};

class TSELoss : public LossFunction, public BasicFunction {
private:
    vector<vector<double>> target;  // [m, b]
    vector<vector<double>> input;   // [m, b]

public:
    void set_target(const vector<vector<double>>& target) {
        this->target = target;
    }

    double get_loss() {
        return output[0][0];
    }

    void forward(const vector<vector<double>>& input) {
        this->input = input;
        tseloss(input, target, output);
    }

    void backward(const vector<vector<double>>& grad_output) {
        tseloss_back(input, target, grad_input);
    }
};
