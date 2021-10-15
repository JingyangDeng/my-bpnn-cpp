#ifndef _PROPAGATION_H
#define _PROPAGATION_H

#include <vector>

double norm(const std::vector<std::vector<double>>& grad_weight);

void update(std::vector<std::vector<double>>& weight, const std::vector<std::vector<double>>& grad_weight, double lr);

void linear(const std::vector<std::vector<double>>& w, const std::vector<std::vector<double>>& x,
            std::vector<std::vector<double>>& output);

void linear_back(const std::vector<std::vector<double>>& grad_output, const std::vector<std::vector<double>>& input,
                 const std::vector<std::vector<double>>& weight, std::vector<std::vector<double>>& grad_input,
                 std::vector<std::vector<double>>& grad_weight);

void sigmoid(const std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& output);

void sigmoid_back(const std::vector<std::vector<double>>& grad_output, const std::vector<std::vector<double>>& output,
                  std::vector<std::vector<double>>& grad_input);

void relu(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output);

void relu_back(const std::vector<std::vector<double>> grad_output, const std::vector<std::vector<double>>& input,
               std::vector<std::vector<double>>& grad_input);

void tseloss(const std::vector<std::vector<double>>& y, const std::vector<std::vector<double>>& t,
             std::vector<std::vector<double>>& output);

void tseloss_back(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                  std::vector<std::vector<double>>& output);

#endif
