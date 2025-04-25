import torch
import torch.nn as nn
from z3 import *
import numpy as np
import time

class DecisionTreeNN(nn.Module):
    """
    A simple neural network designed to mimic a decision tree.
    Basically an AND decision (jumping of XOR experiment):
      - If both inputs are >= 0, then predict class 1.
      - Otherwise, predict class 0.
    """
    def __init__(self):
        super(DecisionTreeNN, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 2, bias=True)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[100.0, 0.0],
                                                  [0.0, 100.0]]))
            self.fc1.bias.copy_(torch.tensor([0.0, 0.0]))
            self.fc2.weight.copy_(torch.tensor([[0.0, 0.0],
                                                  [1.0, 1.0]]))
            self.fc2.bias.copy_(torch.tensor([0.0, -70.0]))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights_and_biases(self):
        params = []
        for layer in [self.fc1, self.fc2]:
            weights = layer.weight.data.numpy()
            biases = layer.bias.data.numpy()
            params.append({'weights': weights, 'biases': biases})
        return params

def encode_nn_in_z3(model, input_vars):
    """
    Encodes a neural network (using get_weights_and_biases) into Z3 constraints.
    """
    params = model.get_weights_and_biases()
    constraints = []
    current_vars = input_vars
    for i, param in enumerate(params):
        weights = param['weights']
        biases = param['biases']
        layer_out = []
        for j in range(len(biases)):
            weighted_sum = biases[j]
            for k in range(len(current_vars)):
                weighted_sum += weights[j, k] * current_vars[k]
            if i < len(params) - 1:
                relu_out = Real(f'relu_{i}_{j}')
                layer_out.append(relu_out)
                constraints.append(relu_out >= 0)
                constraints.append(relu_out >= weighted_sum)
                constraints.append(relu_out == If(weighted_sum > 0, weighted_sum, 0))
            else:
                output_var = Real(f'output_{j}')
                layer_out.append(output_var)
                constraints.append(output_var == weighted_sum)
        current_vars = layer_out
    return constraints, current_vars

def verify_robustness(model, x_sample, epsilon, expected_class):
    """
    Verify the robustness for a given input sample using SMT.
    Returns:
       is_robust (True if no adversarial perturbation found),
       counterexample (if found),
       perturbed_class (classification of perturbed sample).
    """
    start_time = time.time()
    solver = Solver()
    input_dim = len(x_sample)
    x_vars = [Real(f'x_{i}') for i in range(input_dim)]
    x_perturbed = [Real(f'x_perturbed_{i}') for i in range(input_dim)]

    for i in range(input_dim):
        solver.add(x_vars[i] == float(x_sample[i]))
        solver.add(x_perturbed[i] >= float(x_sample[i]) - epsilon)
        solver.add(x_perturbed[i] <= float(x_sample[i]) + epsilon)

    orig_constraints, orig_outputs = encode_nn_in_z3(model, x_vars)
    pert_constraints, pert_outputs = encode_nn_in_z3(model, x_perturbed)

    for c in orig_constraints:
        solver.add(c)
    for c in pert_constraints:
        solver.add(c)

    if expected_class == 0:
        solver.add(pert_outputs[1] > pert_outputs[0])
    else:
        solver.add(pert_outputs[0] > pert_outputs[1])

    result = solver.check()
    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds")

    if result == sat:
        model_solution = solver.model()
        counterexample = np.array([float(model_solution.eval(var).as_decimal(10).replace('?', ''))
                                     for var in x_perturbed])
        perturbed_vals = []
        for out in pert_outputs:
            val = float(model_solution.eval(out).as_decimal(10).replace('?', ''))
            perturbed_vals.append(val)
        pert_class = int(np.argmax(perturbed_vals))
        print("Adversarial example found:")
        print(f"  - Original class: {expected_class}, Perturbed class: {pert_class}")
        print(f"  - Perturbed input: {counterexample}")
        print(f"  - Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        return False, counterexample, pert_class
    else:
        print("No adversarial example found within epsilon bounds.")
        return True, None, None

def gradient_attack_dt(model, x_sample, epsilon, expected_class, steps=100):
    """
    Perform a FGSM-like gradient adversarial attack on the DecisionTreeNN.
    """
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0)
    x_tensor.requires_grad = True
    target = torch.LongTensor([expected_class])
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        logits = model(x_tensor)
        loss = -criterion(logits, target)
        loss.backward()
        grad_sign = x_tensor.grad.sign()
        step_size = epsilon / steps
        x_tensor = x_tensor + step_size * grad_sign
        x_tensor = torch.max(torch.min(x_tensor, torch.FloatTensor(x_sample + epsilon)),
                              torch.FloatTensor(x_sample - epsilon))
        x_tensor = x_tensor.detach().clone().requires_grad_(True)
        pred = model(x_tensor).argmax(dim=1).item()
        if pred != expected_class:
            print(f"Gradient attack succeeded at step {step}")
            return x_tensor.detach().numpy().squeeze(), pred
    return x_tensor.detach().numpy().squeeze(), model(x_tensor).argmax(dim=1).item()

if __name__ == '__main__':
    model = DecisionTreeNN()
    num_samples = 2
    # Generate x random samples in 2 dimensions from uniform distribution on [-1, 1]
    test_samples = [np.random.uniform(-1, 1, 2) for _ in range(num_samples)]
    # Test 5 epsilon values between 0.01 and 0.5
    epsilons = np.linspace(0.01, 0.5, 5)

    for eps in epsilons:
        print(f"\nTesting epsilon = {eps:.3f}")
        smt_vuln = 0
        grad_vuln = 0
        for i, sample in enumerate(test_samples):
            # Expected class: 1 if both features >= 0; else 0.
            expected_class = 1 if (sample[0] >= 0 and sample[1] >= 0) else 0
            print(f"Sample {i+1}: {sample}, Expected Class: {expected_class}")

            print("SMT-based robustness verification:")
            is_robust, counterexample, pert_class = verify_robustness(model, sample, epsilon=eps, expected_class=expected_class)
            if not is_robust:
                smt_vuln += 1

            print("Gradient-based adversarial attack:")
            adv_sample, adv_pred = gradient_attack_dt(model, sample, epsilon=eps, expected_class=expected_class)
            if adv_pred != expected_class:
                grad_vuln += 1

            print(f"Result: SMT - {'Vulnerable' if not is_robust else 'Robust'}, "
                  f"Gradient Attack - {'Vulnerable' if adv_pred != expected_class else 'Robust'}")
            if (not is_robust and adv_pred != expected_class):
                print("Both methods agree: Vulnerable.")
            elif is_robust and adv_pred == expected_class:
                print("Both methods agree: Robust.")
            else:
                print("Methods disagree on the adversarial result!")
            print("-" * 40)
        print(f"Epsilon {eps:.3f}: SMT Vulnerable {smt_vuln}/{num_samples}, Gradient Vulnerable {grad_vuln}/{num_samples}")

