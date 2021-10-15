#include "propagation.h"

#include <cmath>

double norm(const std::vector<std::vector<double>>& grad_weight) {
    double ret = 0;
    for (int i = 0; i < (int)grad_weight.size(); i++) {
        for (int j = 0; j < (int)grad_weight[0].size(); j++) {
            ret += grad_weight[i][j] * grad_weight[i][j];
        }
    }
    return sqrt(ret);
}

void update(std::vector<std::vector<double>>& weight, const std::vector<std::vector<double>>& grad_weight, double lr) {
    for (int i = 0; i < (int)weight.size(); i++) {
        for (int j = 0; j < (int)weight[0].size(); j++) {
            weight[i][j] -= lr * grad_weight[i][j];
        }
    }
}

void linear(const std::vector<std::vector<double>>& w, const std::vector<std::vector<double>>& x,
            std::vector<std::vector<double>>& output) {
    output.resize(w.size(), std::vector<double>(x[0].size()));
    for (int i = 0; i < (int)w.size(); i++) {
        for (int j = 0; j < (int)x[0].size(); j++) {
            double entry = 0;
            for (int k = 0; k < (int)x.size(); k++) {
                entry += w[i][k + 1] * x[k][j];
            }
            entry -= w[i][0];
            output[i][j] = entry;
        }
    }
}

void linear_back(const std::vector<std::vector<double>>& grad_output, const std::vector<std::vector<double>>& input,
                 const std::vector<std::vector<double>>& weight, std::vector<std::vector<double>>& grad_input,
                 std::vector<std::vector<double>>& grad_weight) {
    // grad_weight -- grad_output input
    grad_weight.resize(weight.size(), std::vector<double>(weight[0].size()));
    for (int i = 0; i < (int)weight.size(); i++) {
        double entry = 0;
        for (int k = 0; k < (int)input[0].size(); k++) {
            entry -= grad_output[i][k];
        }
        grad_weight[i][0] = entry;
        for (int j = 0; j < (int)weight[0].size() - 1; j++) {
            double entry = 0;
            for (int k = 0; k < (int)input[0].size(); k++) {
                entry += grad_output[i][k] * input[j][k];
            }
            grad_weight[i][j + 1] = entry;
        }
    }

    // grad_input -- grad_output weight
    grad_input.resize(weight[0].size() - 1, std::vector<double>(input[0].size()));
    for (int i = 0; i < (int)weight[0].size() - 1; i++) {
        for (int j = 0; j < (int)input[0].size(); j++) {
            double entry = 0;
            for (int k = 0; k < (int)weight.size(); k++) {
                entry += grad_output[k][j] * weight[k][i + 1];
            }
            grad_input[i][j] = entry;
        }
    }
}

void sigmoid(const std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& output) {
    output.resize(x.size(), std::vector<double>(x[0].size()));
    for (int i = 0; i < (int)x.size(); i++) {
        for (int j = 0; j < (int)x[0].size(); j++) {
            output[i][j] = 1. / (1. + exp(-x[i][j]));
        }
    }
}

void sigmoid_back(const std::vector<std::vector<double>>& grad_output, const std::vector<std::vector<double>>& output,
                  std::vector<std::vector<double>>& grad_input) {
    grad_input.resize(grad_output.size(), std::vector<double>(grad_output[0].size()));
    for (int i = 0; i < (int)grad_output.size(); i++) {
        for (int j = 0; j < (int)grad_output[0].size(); j++) {
            grad_input[i][j] = grad_output[i][j] * output[i][j] * (1 - output[i][j]);
        }
    }
}

void relu(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output) {
    output.resize(input.size(), std::vector<double>(input[0].size()));
    for (int i = 0; i < (int)input.size(); i++) {
        for (int j = 0; j < (int)input[0].size(); j++) {
            output[i][j] = input[i][j] < 0 ? 0 : input[i][j];
        }
    }
}

void relu_back(const std::vector<std::vector<double>> grad_output, const std::vector<std::vector<double>>& input,
               std::vector<std::vector<double>>& grad_input) {
    grad_input.resize(grad_output.size(), std::vector<double>(grad_output[0].size()));
    for (int i = 0; i < (int)grad_output.size(); i++) {
        for (int j = 0; j < (int)grad_output[0].size(); j++) {
            grad_input[i][j] = input[i][j] < 0 ? 0 : grad_output[i][j];
        }
    }
}

void tseloss(const std::vector<std::vector<double>>& y, const std::vector<std::vector<double>>& t,
             std::vector<std::vector<double>>& output) {
    output.resize(1, std::vector<double>(1));
    output[0][0] = 0;
    for (int i = 0; i < (int)y.size(); i++) {
        for (int j = 0; j < (int)y[0].size(); j++) {
            output[0][0] += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
        }
    }
    output[0][0] *= 0.5;
}

void tseloss_back(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                  std::vector<std::vector<double>>& output) {
    output.resize(x.size(), std::vector<double>(x[0].size()));
    for (int i = 0; i < (int)x.size(); i++) {
        for (int j = 0; j < (int)x[0].size(); j++) {
            output[i][j] = x[i][j] - y[i][j];
        }
    }
}
