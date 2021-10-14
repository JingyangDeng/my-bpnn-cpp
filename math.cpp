#include <cmath>
#include <vector>
using namespace std;

double norm(const vector<vector<double>>& grad_weight) {
    double ret = 0;
    for (int i = 0; i < (int)grad_weight.size(); i++) {
        for (int j = 0; j < (int)grad_weight[0].size(); j++) {
            ret += grad_weight[i][j] * grad_weight[i][j];
        }
    }
    return sqrt(ret);
}

void update(vector<vector<double>>& weight, const vector<vector<double>>& grad_weight, double lr) {
    for (int i = 0; i < (int)weight.size(); i++) {
        for (int j = 0; j < (int)weight[0].size(); j++) {
            weight[i][j] -= lr * grad_weight[i][j];
        }
    }
}

void linear(const vector<vector<double>>& w, const vector<vector<double>>& x, vector<vector<double>>& output) {
    output.resize(w.size(), vector<double>(x[0].size()));
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

void linear_back(const vector<vector<double>>& grad_output, const vector<vector<double>>& input,
                 const vector<vector<double>>& weight, vector<vector<double>>& grad_input,
                 vector<vector<double>>& grad_weight) {
    // grad_weight -- grad_output input
    grad_weight.resize(weight.size(), vector<double>(weight[0].size()));
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
    grad_input.resize(weight[0].size() - 1, vector<double>(input[0].size()));
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

void sigmoid(const vector<vector<double>>& x, vector<vector<double>>& output) {
    output.resize(x.size(), vector<double>(x[0].size()));
    for (int i = 0; i < (int)x.size(); i++) {
        for (int j = 0; j < (int)x[0].size(); j++) {
            output[i][j] = 1. / (1. + exp(-x[i][j]));
        }
    }
}

void sigmoid_back(const vector<vector<double>>& grad_output, const vector<vector<double>>& output,
                  vector<vector<double>>& grad_input) {
    grad_input.resize(grad_output.size(), vector<double>(grad_output[0].size()));
    for (int i = 0; i < (int)grad_output.size(); i++) {
        for (int j = 0; j < (int)grad_output[0].size(); j++) {
            grad_input[i][j] = grad_output[i][j] * output[i][j] * (1 - output[i][j]);
        }
    }
}

void relu(const vector<vector<double>>& input, vector<vector<double>>& output) {
    output.resize(input.size(), vector<double>(input[0].size()));
    for (int i = 0; i < (int)input.size(); i++) {
        for (int j = 0; j < (int)input[0].size(); j++) {
            output[i][j] = input[i][j] < 0 ? 0 : input[i][j];
        }
    }
}

void relu_back(const vector<vector<double>> grad_output, const vector<vector<double>>& input,
               vector<vector<double>>& grad_input) {
    grad_input.resize(grad_output.size(), vector<double>(grad_output[0].size()));
    for (int i = 0; i < (int)grad_output.size(); i++) {
        for (int j = 0; j < (int)grad_output[0].size(); j++) {
            grad_input[i][j] = input[i][j] < 0 ? 0 : grad_output[i][j];
        }
    }
}

void tseloss(const vector<vector<double>>& y, const vector<vector<double>>& t, vector<vector<double>>& output) {
    output.resize(1, vector<double>(1));
    output[0][0] = 0;
    for (int i = 0; i < (int)y.size(); i++) {
        for (int j = 0; j < (int)y[0].size(); j++) {
            output[0][0] += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
        }
    }
    output[0][0] *= 0.5;
}

void tseloss_back(const vector<vector<double>>& x, const vector<vector<double>>& y, vector<vector<double>>& output) {
    output.resize(x.size(), vector<double>(x[0].size()));
    for (int i = 0; i < (int)x.size(); i++) {
        for (int j = 0; j < (int)x[0].size(); j++) {
            output[i][j] = x[i][j] - y[i][j];
        }
    }
}
