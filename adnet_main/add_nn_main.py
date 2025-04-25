import torch
import torch.nn as nn
import json
from pathlib import Path
from safetensors.torch import safe_open
from z3 import *
import numpy as np
import time

class Adnet(nn.Module):
    """
    If the input changes by at most ε, does the output stay within δ of the original?
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Adnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_weights_and_biases(model):
        return [
            {'weights': model.fc1.weight.detach().numpy(), 'biases': model.fc1.bias.detach().numpy()},
            {'weights': model.fc2.weight.detach().numpy(), 'biases': model.fc2.bias.detach().numpy()},
            {'weights': model.fc3.weight.detach().numpy(), 'biases': model.fc3.bias.detach().numpy()},
        ]
    
from z3 import Real, If

def encode_nn_in_z3(model, input_vars):
    """
    Encodes the Adnet model as Z3 constraints.

    Args:
        model: Adnet model with loaded weights.
        input_vars: List of Z3 Real variables (e.g., [x1, x2]).

    Returns:
        constraints: List of Z3 constraints (ReLU + linear ops).
        output_var: Z3 Real variable representing final output.
    """
    constraints = []
    params = model.get_weights_and_biases()
    current_vars = input_vars

    for i, layer in enumerate(params):
        weights = layer["weights"]
        biases = layer["biases"]
        layer_out = []

        for j in range(len(biases)):
            # Compute weighted sum: sum_i w_ji * x_i + b_j
            weighted_sum = RealVal(float(biases[j]))  # make it symbolic
            for k in range(len(current_vars)):
                weight_val = float(weights[j, k])
                weighted_sum += weight_val * current_vars[k]

            if i < len(params) - 1:
                # Apply ReLU
                relu_out = weighted_sum  #Real(f'relu_{i}_{j}')
                layer_out.append(relu_out)
                constraints.append(relu_out >= 0)
                constraints.append(relu_out >= weighted_sum)
                constraints.append(relu_out == If(weighted_sum > 0, weighted_sum, 0))
            else:
                # Final layer: no ReLU
                output_var = Real(f'output_{j}')
                layer_out.append(output_var)
                constraints.append(output_var == weighted_sum)

        current_vars = layer_out

    return constraints, current_vars[0]  # Final output var (assumes 1 output)


def verify_robustness(model, x_sample, epsilon, num_inputs=2):
    """
    Verify whether the neural network is robust to perturbations within epsilon
    for a given input sample.
    
    Returns:
    - is_robust: Boolean indicating whether the model is robust for this sample
    - counterexample: If not robust, a perturbed input that causes misclassification
    """
    start_time = time.time()
    
    # Initialize Z3 solver
    solver = Solver()
    
    # Create variables for the input and perturbed input
    input_dim = x_sample.shape[0]
    x_vars = [Real(f'x_{i}') for i in range(input_dim)]
    x_perturbed = [Real(f'x_perturbed_{i}') for i in range(input_dim)]
    
    # Add constraints for perturbation
    for i in range(input_dim):
        # Original input value
        solver.add(x_vars[i] == x_sample[i].item())
        
        # Perturbation bounds
        solver.add(x_perturbed[i] >= x_sample[i].item() - epsilon)
        solver.add(x_perturbed[i] <= x_sample[i].item() + epsilon)
    
    # Encode the neural network for both original and perturbed inputs
    original_constraints, original_output_vars = encode_nn_in_z3(model, x_vars)
    perturbed_constraints, perturbed_output_vars = encode_nn_in_z3(model, x_perturbed)
    # Add all constraints to the solver
    for constraint in original_constraints:
        solver.add(constraint)
    
    for constraint in perturbed_constraints:
        solver.add(constraint)

    # Add constraint that the perturbed input is |f(x1', x2') - f(x1, x2)| > δ
    solver.add(If(perturbed_output_vars - original_output_vars >= 0,
              perturbed_output_vars - original_output_vars,
              original_output_vars - perturbed_output_vars) < epsilon)
   
    # Check satisfiability
    result = solver.check()
    
    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds")
    
    if result == sat:
        # Found a counterexample (adversarial example)
        model_solution = solver.model()
        print("Output perturbed {model_solution[perturbed_output_vars]} original {model_solution[original_output_vars]}")

        # Extract the perturbed input values
        counterexample = np.array([float(model_solution.eval(var).as_decimal(10).replace('?', '')) 
                                for var in x_perturbed])
        
        # Find predicted class for perturbed input
        perturbed_outputs = []
        
        for j in range(num_inputs):
            # Get the output variable's value from the model
            value = float(model_solution.eval(perturbed_output_vars[j]).as_decimal(10).replace('?', ''))
            perturbed_outputs.append(value)
        
        print(f"Found adversarial example: Original output: {model_solution[original_output_vars]}, Perturbed output: {model_solution[perturbed_output_vars]}")
        print(f"Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        
        return False, counterexample, model_solution[perturbed_output_vars]
    else:
        print(f"No adversarial example found within epsilon = {epsilon}")
        return True, None, None
    
import torch
import torch.nn.functional as F

def gradient_attack_regression(model, x, epsilon, step_size=0.01, num_steps=20, delta_thresh=0.1):
    """
    Performs a gradient-based adversarial attack on a regression model.
    
    Args:
        model: The Adnet model (output: scalar).
        x: Input tensor (shape: [input_dim]).
        epsilon: Max L-infinity perturbation allowed.
        step_size: Step size for gradient update.
        num_steps: Number of steps in the attack.
        delta_thresh: The minimum delta to trigger a successful attack.

    Returns:
        x_adv: Adversarial input.
        f(x_adv): Output of model on adversarial input.
        delta: Absolute difference from original output.
    """
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    orig_output = model(x).item()

    x_adv = x.clone()

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        output = model(x_adv)[0]

        # Goal: maximize |f(x_adv) - f(x)|
        loss = -torch.abs(output - orig_output)
        loss.backward()

        # Gradient ascent step
        grad = x_adv.grad.detach()
        x_adv = x_adv + step_size * torch.sign(grad)

        # Project back into ε-ball (L-inf constraint)
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

    final_output = model(x_adv).item()
    delta = abs(final_output - orig_output)

    return x_adv, final_output, delta


def compare_verification_methods(model, x_sample, epsilon, prediction):
    """Compare SMT-based verification with gradient-based attacks."""
    print("\n=== SMT-based Verification ===")
    is_robust_smt, counterexample_smt, perturbed_output_smt = verify_robustness(
        model, x_sample, epsilon
    )
    
    print("\n=== Gradient-based Attack ===")
    x_perturbed_grad, perturbed_grad, delta_grad = gradient_attack_regression(
        model, x_sample, epsilon, delta_thresh=0.1
    )
    
    print("\n=== Comparison ===")
    if is_robust_smt:
        print("SMT verification found no adversarial examples within epsilon boundaries")
        print(prediction, perturbed_grad)
        if round(prediction.item(),3) != round(perturbed_grad,3):
            print("WARNING: Gradient-based attack found an adversarial example when SMT didn't")
            print("This may indicate an issue with the SMT encoding or epsilon handling")
        else:
            print("Gradient-based attack also failed to find adversarial examples - models agree")
    else:
        print("SMT verification found an adversarial example:")
        print(f"  - Perturbed class: {perturbed_output_smt}")
        print(f"  - Perturbation magnitude: {np.linalg.norm(counterexample_smt - x_sample)}")
        
        if round(perturbed_grad,3) != round(prediction.item(),3):
            print("Gradient-based attack also found an adversarial example:")
            print(f"  - Perturbed class: {perturbed_grad}")
            print(f"  - Perturbation magnitude: {np.linalg.norm(x_perturbed_grad - x_sample)}")
        else:
            print("Gradient-based attack failed to find an adversarial example")
            print("This could indicate the SMT solver is more powerful for this case")
    
    return is_robust_smt, counterexample_smt, perturbed_output_smt, x_perturbed_grad, perturbed_grad

# Load config.json
with open("adnet/config.json") as f:
    config = json.load(f)

# Initialize the model
model = Adnet(
    input_size=config["input_size"],
    hidden_size1=config["hidden_size1"],
    hidden_size2=config["hidden_size2"],
    output_size=config["output_size"]
)

# Load the weights
weights_path = "adnet/model.safetensors"
state_dict = {}
with safe_open(weights_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

# Load into model
model.load_state_dict(state_dict)
model.eval()

x_sample = torch.tensor([10.111, 12.318], dtype=torch.float32)  # shape (2,)

with torch.no_grad():
    prediction = model(x_sample)

print(f"Sample: {x_sample}")
print(f"Predicted class: {prediction}")

# Verify robustness with different epsilon values
for epsilon in [0.001, 0.01, 0.1]:
    print(f"\n=== Robustness Verification (Epsilon = {epsilon}) ===")
    is_robust_smt, counterexample_smt, perturbed_output_smt, x_perturbed_grad, perturbed_grad = \
        compare_verification_methods(model, x_sample, epsilon, prediction)
    
    if not is_robust_smt:
        print("\nFound adversarial example using SMT:")
        print(f"Original input: {x_sample}")
        print(f"Perturbed input: {counterexample_smt}")
        print(f"Perturbation: {counterexample_smt - x_sample}")
        print(f"Original class: {prediction}, Perturbed class: {perturbed_output_smt}")

print("\nCheckpoint 3 completed!")