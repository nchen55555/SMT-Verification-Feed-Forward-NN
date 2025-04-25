import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from z3 import Real, If, Solver, sat
import torch.nn.functional as F


class MinMaxDecisionNet(nn.Module):
    """
    A fixed network that mimics a min-max decision.
    Given an input vector of 3 features: [x1, x2, x3],
    it computes a nested maximum:
       s = max(x2, x3) = ReLU(x2 - x3) + x3
       m = max(x1, s) = ReLU(x1 - s) + s
    Final logits: [ -m, m ], so that if m > 0, class 1 is predicted.
    """
    def __init__(self):
        super(MinMaxDecisionNet, self).__init__()
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
        t1 = x[:, 1] - x[:, 2]
        r = F.relu(t1)
        s = r + x[:, 2]
        t2 = x[:, 0] - s
        r2 = F.relu(t2)
        m = r2 + s
        out0 = -m
        out1 = m
        logits = torch.stack((out0, out1), dim=1)
        return logits
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

def encode_minmax_nn_in_z3(input_vars):
    """
    Given a list of Z3 Real variables for [x1, x2, x3],
    encode the fixed computation of the MinMaxDecisionNet.
    Returns:
       constraints: list of Z3 constraints
       outputs: list [o0, o1]
    """
    constraints = []
    x1, x2, x3 = input_vars
    t1 = x2 - x3
    r = Real('r')
    constraints.append(r >= 0)
    constraints.append(r >= t1)
    constraints.append(r == If(t1 > 0, t1, 0))
    s = r + x3
    t2 = x1 - s
    r2 = Real('r2')
    constraints.append(r2 >= 0)
    constraints.append(r2 >= t2)
    constraints.append(r2 == If(t2 > 0, t2, 0))
    m = r2 + s
    o0 = Real('o0')
    o1 = Real('o1')
    constraints.append(o0 == -m)
    constraints.append(o1 == m)
    return constraints, [o0, o1]

def verify_robustness_minmax(model, x_sample, epsilon, expected_class):
    """
    For a given sample x_sample (array of 3 values),
    verify whether a small perturbation (within L∞ epsilon)
    can cause a misclassification.
    Returns:
        is_robust: True if no adversarial perturbation is found,
        counterexample: the perturbed input (if found),
        perturbed_class: predicted class for the perturbed input.
    """
    start_time = time.time()
    solver = Solver()
    input_dim = len(x_sample)
    x_vars = [Real(f'x_{i}') for i in range(input_dim)]
    x_pert = [Real(f'x_pert_{i}') for i in range(input_dim)]
    for i in range(input_dim):
        solver.add(x_vars[i] == float(x_sample[i]))
        solver.add(x_pert[i] >= float(x_sample[i]) - epsilon)
        solver.add(x_pert[i] <= float(x_sample[i]) + epsilon)
    orig_constraints, orig_outputs = encode_minmax_nn_in_z3(x_vars)
    pert_constraints, pert_outputs = encode_minmax_nn_in_z3(x_pert)
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
    print(f"SMT Verification completed in {end_time - start_time:.2f} seconds")
    if result == sat:
        model_sol = solver.model()
        counterexample = np.array([float(model_sol.eval(var).as_decimal(10).replace('?', ''))
                                     for var in x_pert])
        pert_outs = [float(model_sol.eval(out).as_decimal(10).replace('?', ''))
                     for out in pert_outputs]
        pert_class = int(np.argmax(pert_outs))
        print("Adversarial example found:")
        print(f"  Original sample: {x_sample}")
        print(f"  Perturbed sample: {counterexample}")
        print(f"  Original expected class: {expected_class}, Perturbed class: {pert_class}")
        print(f"  Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        return False, counterexample, pert_class
    else:
        print("No adversarial example found within epsilon bounds.")
        return True, None, None

def gradient_attack_minmax(model, x_sample, epsilon, expected_class, steps=100):
    """
    Perform a FGSM-like gradient adversarial attack on the fixed network.
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
    model = MinMaxDecisionNet()
    num_samples = 5
    # Generate x random samples in 3 dimensions from a uniform distribution on [-1, 1]
    test_samples = [np.random.uniform(-1, 1, 3) for _ in range(num_samples)]
    # Test y epsilon values between 0.01 and 0.5
    epsilons = np.linspace(0.01, 0.5, 5)

    for eps in epsilons:
        print(f"\nTesting epsilon = {eps:.3f}")
        smt_vuln = 0
        grad_vuln = 0
        for i, sample in enumerate(test_samples):
            sample_tensor = torch.FloatTensor(sample).unsqueeze(0)
            with torch.no_grad():
                exp_class = int(model.predict(sample_tensor).item())
            print(f"Sample {i+1}: {sample}, Expected Class: {exp_class}")

            print("SMT-based robustness verification:")
            is_robust, counterexample, pert_class = verify_robustness_minmax(model, sample, epsilon=eps, expected_class=exp_class)
            if not is_robust:
                print("Result: SMT found an adversarial example.")
                print(f"Perturbed sample: {counterexample}, New Prediction: {pert_class}")
                smt_vuln += 1
            else:
                print("Result: SMT did not find an adversarial example.")

            print("Gradient-based adversarial attack:")
            adv_sample, adv_pred = gradient_attack_minmax(model, sample, epsilon=eps, expected_class=exp_class)
            if adv_pred != exp_class:
                print("Result: Gradient attack found an adversarial example.")
                print(f"Perturbed sample: {adv_sample}, New Prediction: {adv_pred}")
                grad_vuln += 1
            else:
                print("Result: Gradient attack did not find an adversarial example.")

            if (not is_robust and adv_pred != exp_class):
                print("Both methods agree: Vulnerable.")
            elif is_robust and adv_pred == exp_class:
                print("Both methods agree: Robust.")
            else:
                print("Methods disagree on the adversarial result!")

            print("-" * 40)
        print(f"Epsilon {eps:.3f}: SMT Vulnerable {smt_vuln}/{num_samples}, Gradient Vulnerable {grad_vuln}/{num_samples}")
