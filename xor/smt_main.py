import numpy as np
import torch
import torch.nn as nn
from z3 import *
import time

class SimpleNN(nn.Module):
    """A simple neural network with configurable architecture."""
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
        """Extract weights and biases from the model."""
        params = []
        for layer in self.layers:
            weights = layer.weight.data.numpy()
            biases = layer.bias.data.numpy()
            params.append({'weights': weights, 'biases': biases})
        return params

def train_simple_model(input_size, hidden_sizes, output_size, X_train, y_train, epochs=1000, lr=0.01):
    """Train a simple neural network."""
    model = SimpleNN(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def save_model_to_files(model, params_file, architecture_file):
    """Save model parameters and architecture to files."""
    # Save parameters as a list of dictionaries
    params = []
    for layer in model.layers:
        layer_params = {
            'weights': layer.weight.data.numpy(),
            'biases': layer.bias.data.numpy()
        }
        params.append(layer_params)
    
    np.save(params_file, np.array(params, dtype=object))
    
    # Save architecture
    architecture = {
        'input_size': model.layers[0].weight.shape[1],
        'hidden_sizes': [layer.weight.shape[0] for layer in model.layers[:-1]],
        'output_size': model.layers[-1].weight.shape[0]
    }
    
    with open(architecture_file, 'w') as f:
        f.write(str(architecture))

def load_model_from_files(params_file, architecture_file):
    """Load a neural network model from parameter and architecture files."""
    # Load architecture
    with open(architecture_file, 'r') as f:
        arch = eval(f.read())
    
    input_size = arch['input_size']
    hidden_sizes = arch['hidden_sizes']
    output_size = arch['output_size']
    
    # Create model
    model = SimpleNN(input_size, hidden_sizes, output_size)
    
    # Load parameters
    params = np.load(params_file, allow_pickle=True)
    
    # Assign parameters to model
    for i, layer in enumerate(model.layers):
        layer_params = params[i]
        layer.weight.data = torch.FloatTensor(layer_params['weights'])
        layer.bias.data = torch.FloatTensor(layer_params['biases'])
    
    return model

def encode_nn_in_z3(model, input_vars):
    """Encode a neural network as Z3 constraints."""
    params = model.get_weights_and_biases()
    layer_outputs = []
    current_vars = input_vars
    constraints = []
    
    for i, param_dict in enumerate(params):
        weights = param_dict['weights']
        biases = param_dict['biases']
        layer_out = []
        
        for j in range(len(biases)):
            # Compute weighted sum + bias
            weighted_sum = biases[j]
            for k in range(len(current_vars)):
                weighted_sum += weights[j, k] * current_vars[k]
            
            # Apply ReLU for all but the last layer
            if i < len(params) - 1:
                relu_out = Real(f'relu_{i}_{j}')
                # ReLU(x) = max(0, x)
                layer_out.append(relu_out)
                # Add constraint: relu_out >= 0
                constraints.append(relu_out >= 0)
                # Add constraint: relu_out >= weighted_sum
                constraints.append(relu_out >= weighted_sum)
                # Add constraint: (relu_out == 0) OR (relu_out == weighted_sum)
                constraints.append(relu_out == If(weighted_sum > 0, weighted_sum, 0))
            else:
                # For the output layer, no ReLU
                output_var = Real(f'output_{j}')
                layer_out.append(output_var)
                constraints.append(output_var == weighted_sum)
        
        layer_outputs.append(layer_out)
        current_vars = layer_out
    
    return constraints, layer_outputs[-1]  # Return both constraints and the final layer output variables

def verify_robustness(model, x_sample, epsilon, expected_class, num_classes):
    """
    Verify whether the neural network is robust to perturbations within epsilon
    for a given input sample.
    
    Returns:
    - is_robust: Boolean indicating whether the model is robust for this sample
    - counterexample: If not robust, a perturbed input that causes misclassification
    - perturbed_class: The class predicted for the perturbed input
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
        solver.add(x_vars[i] == x_sample[i])
        
        # Perturbation bounds
        solver.add(x_perturbed[i] >= x_sample[i] - epsilon)
        solver.add(x_perturbed[i] <= x_sample[i] + epsilon)
    
    # Encode the neural network for both original and perturbed inputs
    original_constraints, original_output_vars = encode_nn_in_z3(model, x_vars)
    perturbed_constraints, perturbed_output_vars = encode_nn_in_z3(model, x_perturbed)
    
    # Add all constraints to the solver
    for constraint in original_constraints:
        solver.add(constraint)
    
    for constraint in perturbed_constraints:
        solver.add(constraint)
    
    # Add constraint that the perturbed input leads to a different class
    for i in range(num_classes):
        if i != expected_class:
            # The output for another class is greater than the expected class
            solver.add(perturbed_output_vars[i] > perturbed_output_vars[expected_class])
            break  # Only need to find one misclassification
    
    # Check satisfiability
    result = solver.check()
    
    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds")
    
    if result == sat:
        # Found a counterexample (adversarial example)
        model_solution = solver.model()
        
        # Extract the perturbed input values
        counterexample = np.array([float(model_solution.eval(var).as_decimal(10).replace('?', '')) 
                                for var in x_perturbed])
        
        # Find predicted class for perturbed input
        perturbed_outputs = []
        
        for j in range(num_classes):
            # Get the output variable's value from the model
            value = float(model_solution.eval(perturbed_output_vars[j]).as_decimal(10).replace('?', ''))
            perturbed_outputs.append(value)
        
        perturbed_class = np.argmax(perturbed_outputs)
        
        print(f"Found adversarial example: Original class: {expected_class}, Perturbed class: {perturbed_class}")
        print(f"Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        
        return False, counterexample, perturbed_class
    else:
        print(f"No adversarial example found within epsilon = {epsilon}")
        return True, None, None

def adversarial_attack(model, x_sample, epsilon, expected_class, steps=100):
    """
    Perform a simple gradient-based adversarial attack (FGSM-like)
    to compare with SMT results.
    """
    # Convert to PyTorch tensor
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0)
    x_tensor.requires_grad = True
    
    # Target is the expected class
    target = torch.LongTensor([expected_class])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Maximize loss instead of minimizing
    for i in range(steps):
        # Forward pass
        output = model(x_tensor)
        loss = -criterion(output, target)  # Negative because we want to maximize
        
        # Backward pass
        loss.backward()
        
        # Get gradient sign
        grad_sign = x_tensor.grad.sign().detach().numpy().squeeze()
        
        # Update tensor
        step_size = epsilon / steps
        x_tensor_np = x_tensor.detach().numpy().squeeze()
        x_tensor_np += step_size * grad_sign
        
        # Clip to ensure we're within epsilon
        x_tensor_np = np.clip(x_tensor_np, x_sample - epsilon, x_sample + epsilon)
        
        # Create new tensor
        x_tensor = torch.FloatTensor(x_tensor_np).unsqueeze(0)
        x_tensor.requires_grad = True
        
        # Get predictions
        pred = model(x_tensor).detach().numpy().squeeze()
        pred_class = np.argmax(pred)
        
        # Check if we've successfully created an adversarial example
        if pred_class != expected_class:
            print(f"Found adversarial example at step {i}")
            return x_tensor_np, pred_class
    
    # Get final prediction
    pred = model(x_tensor).detach().numpy().squeeze()
    pred_class = np.argmax(pred)
    
    return x_tensor.detach().numpy().squeeze(), pred_class

def compare_verification_methods(model, x_sample, epsilon, expected_class, num_classes):
    """Compare SMT-based verification with gradient-based attacks."""
    print("\n=== SMT-based Verification ===")
    is_robust_smt, counterexample_smt, perturbed_class_smt = verify_robustness(
        model, x_sample, epsilon, expected_class, num_classes
    )
    
    print("\n=== Gradient-based Attack ===")
    x_perturbed_grad, perturbed_class_grad = adversarial_attack(
        model, x_sample, epsilon, expected_class
    )
    
    print("\n=== Comparison ===")
    if is_robust_smt:
        print("SMT verification found no adversarial examples within epsilon boundaries")
        if perturbed_class_grad != expected_class:
            print("WARNING: Gradient-based attack found an adversarial example when SMT didn't")
            print("This may indicate an issue with the SMT encoding or epsilon handling")
        else:
            print("Gradient-based attack also failed to find adversarial examples - models agree")
    else:
        print("SMT verification found an adversarial example:")
        print(f"  - Perturbed class: {perturbed_class_smt}")
        print(f"  - Perturbation magnitude: {np.linalg.norm(counterexample_smt - x_sample)}")
        
        if perturbed_class_grad != expected_class:
            print("Gradient-based attack also found an adversarial example:")
            print(f"  - Perturbed class: {perturbed_class_grad}")
            print(f"  - Perturbation magnitude: {np.linalg.norm(x_perturbed_grad - x_sample)}")
        else:
            print("Gradient-based attack failed to find an adversarial example")
            print("This could indicate the SMT solver is more powerful for this case")
    
    return is_robust_smt, counterexample_smt, perturbed_class_smt, x_perturbed_grad, perturbed_class_grad

def main():
    # Example usage
    
    # 1. Create or load a small dataset
    # For demonstration, let's create a simple XOR-like dataset
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    
    y_train = np.array([0, 1, 1, 0], dtype=np.int64)  # XOR function
    
    # 2. Train a simple neural network
    input_size = 2
    hidden_sizes = [5, 5]  # Small network for demonstration
    output_size = 2  # Binary classification
    
    print("Training neural network...")
    model = train_simple_model(input_size, hidden_sizes, output_size, X_train, y_train)
    
    # 3. Save the model to files
    params_file = 'nn_params.npy'
    architecture_file = 'nn_architecture.txt'
    save_model_to_files(model, params_file, architecture_file)
    
    # 4. Load the model from files (in a real scenario, you'd just load it)
    print("Loading model from files...")
    model = load_model_from_files(params_file, architecture_file)
    
    # 5. Verify robustness for a sample
    x_sample = np.array([0, 1], dtype=np.float32)  # Should be class 1
    
    # Predict using PyTorch
    with torch.no_grad():
        prediction = model(torch.FloatTensor(x_sample))
        predicted_class = torch.argmax(prediction).item()
    
    print(f"Sample: {x_sample}")
    print(f"Predicted class: {predicted_class}")
    
    # 6. Verify robustness with different epsilon values
    for epsilon in [0.1, 0.2, 0.3]:
        print(f"\n=== Robustness Verification (Epsilon = {epsilon}) ===")
        is_robust_smt, counterexample_smt, perturbed_class_smt, x_perturbed_grad, perturbed_class_grad = \
            compare_verification_methods(model, x_sample, epsilon, predicted_class, output_size)
        
        if not is_robust_smt:
            print("\nFound adversarial example using SMT:")
            print(f"Original input: {x_sample}")
            print(f"Perturbed input: {counterexample_smt}")
            print(f"Perturbation: {counterexample_smt - x_sample}")
            print(f"Original class: {predicted_class}, Perturbed class: {perturbed_class_smt}")
    
    print("\nCheckpoint 1 MVP completed!")

if __name__ == "__main__":
    main()
