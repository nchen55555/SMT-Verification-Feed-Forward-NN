import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from z3 import Real, If, Solver, sat
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, output_size))
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    def get_weights_and_biases(self):
        params = []
        for layer in self.layers:
            weights = layer.weight.data.numpy()
            biases = layer.bias.data.numpy()
            params.append({'weights': weights, 'biases': biases})
        return params

def train_simple_model(model, X_train, y_train, epochs=500, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss_history

def encode_nn_in_z3(model, input_vars):
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
    start_time = time.time()
    solver = Solver()
    input_dim = x_sample.shape[0]
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
    print(f"SMT Verification completed in {end_time - start_time:.2f} seconds")
    if result == sat:
        model_solution = solver.model()
        counterexample = np.array([float(model_solution.eval(var).as_decimal(10).replace('?', ''))
                                     for var in x_perturbed])
        pert_outs = [float(model_solution.eval(out).as_decimal(10).replace('?', ''))
                     for out in pert_outputs]
        pert_class = int(np.argmax(pert_outs))
        print("Adversarial example found:")
        print(f"  - Original class: {expected_class}, Perturbed class: {pert_class}")
        print(f"  - Perturbed input: {counterexample}")
        print(f"  - Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        return False, counterexample, pert_class
    else:
        print("No adversarial example found within epsilon bounds.")
        return True, None, None

def gradient_attack(model, x_sample, epsilon, expected_class, steps=100):
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

def plot_decision_boundary_with_epsilon(model, X, y, samples, epsilon, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    with torch.no_grad():
        Z = model(grid_tensor).argmax(dim=1).numpy()
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.Spectral)
    for sample in samples:
        circle = plt.Circle((sample[0], sample[1]), epsilon, color='black', fill=False, linestyle='dashed', linewidth=2)
        ax.add_patch(circle)
        plt.plot(sample[0], sample[1], 'ko', markersize=5)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def run_experiment():
    # Force a more vulnerable scenario:
    # Lower class separation and increase label noise.
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, class_sep=0.05,
                               flip_y=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_size = 2
    hidden_sizes = [10, 10]
    output_size = 2
    model = SimpleNN(input_size, hidden_sizes, output_size)

    print("Training neural network on Two-Gaussians dataset (vulnerable setting)...")
    loss_history = train_simple_model(model, X_train, y_train, epochs=500, lr=0.01)
    plt.figure(figsize=(8,6))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    with torch.no_grad():
        model.layers[-1].weight.data *= 0.1

    print("Plotting decision boundary on training data:")
    plot_decision_boundary_with_epsilon(model, X_train, y_train, samples=[], epsilon=0, title="Training Data Decision Boundary")

    print("Evaluating robustness on test data:")
    num_samples = 20
    epsilons = [0.1, 0.5, 1.0]
    selected_indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    results = []
    for eps in epsilons:
        print(f"\nTesting epsilon = {eps:.3f}")
        smt_vuln = 0
        grad_vuln = 0
        samples_to_plot = []
        for i in selected_indices:
            x_sample = X_test[i]
            sample_tensor = torch.FloatTensor(x_sample).unsqueeze(0)
            with torch.no_grad():
                exp_class = int(model(sample_tensor).argmax(dim=1).item())
            print(f"Sample: {x_sample}, Expected Class: {exp_class}")
            print("SMT-based verification:")
            is_robust, counterexample, pert_class = verify_robustness(model, x_sample, epsilon=eps, expected_class=exp_class)
            if not is_robust:
                smt_vuln += 1
            print("Gradient-based attack:")
            adv_sample, adv_pred = gradient_attack(model, x_sample, epsilon=eps, expected_class=exp_class)
            if adv_pred != exp_class:
                grad_vuln += 1
            print(f"Result: SMT - {'Vulnerable' if not is_robust else 'Robust'}, "
                  f"Gradient Attack - {'Vulnerable' if adv_pred != exp_class else 'Robust'}")
            if (not is_robust and adv_pred != exp_class):
                print("Both methods agree: Vulnerable.")
            elif is_robust and adv_pred == exp_class:
                print("Both methods agree: Robust.")
            else:
                print("Methods disagree on the adversarial result!")
            print("-" * 40)
            if len(samples_to_plot) < 3:
                samples_to_plot.append(x_sample)
        print(f"Epsilon {eps:.3f}: SMT Vulnerable {smt_vuln}/{num_samples}, Gradient Vulnerable {grad_vuln}/{num_samples}")
        results.append({"epsilon": eps, "smt_vulnerable": smt_vuln, "grad_vulnerable": grad_vuln})
        plot_decision_boundary_with_epsilon(model, X_test, y_test, samples_to_plot, eps,
                                              title=f"Decision Boundary with ε-ball (epsilon = {eps:.3f})")

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10,6))
    plt.plot(results_df['epsilon'], results_df['smt_vulnerable'] / num_samples, 'o-', label='SMT Vulnerability Rate')
    plt.plot(results_df['epsilon'], results_df['grad_vulnerable'] / num_samples, 's-', label='Gradient Vulnerability Rate')
    plt.xlabel("Epsilon")
    plt.ylabel("Vulnerability Rate")
    plt.title("Vulnerability Rate vs Epsilon")
    plt.legend()
    plt.grid(True)
    plt.show()
    return results_df, model

if __name__ == '__main__':
    run_experiment()