import numpy as np
import torch
from z3 import *
import time
import torch.nn as nn


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# class FeedForwardNeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         # Weights initialization
#         self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
#         self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

#         # Biases initialization
#         self.bias_hidden = np.random.rand(1, self.hidden_size)
#         self.bias_output = np.random.rand(1, self.output_size)


#     def forward(self, input_data):
#         # Input to hidden layer
#         self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = sigmoid(self.hidden_input)

#         # Hidden layer to output layer
#         self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.final_output = sigmoid(self.final_input)

#         return self.final_output

#     def backward(self, input_data, target_output, learning_rate):
#         # Calculate the error
#         output_error = target_output - self.final_output
#         output_delta = output_error * sigmoid_derivative(self.final_output)

#         # Calculate the hidden layer error
#         hidden_error = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

#         # Update the weights and biases
#         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
#         self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
#         self.weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


#     def train(self, input_data, target_output, epochs, learning_rate):
#         for epoch in range(epochs):
#             # Forward pass
#             self.forward(input_data)

#             # Backward pass and weight updates
#             self.backward(input_data, target_output, learning_rate)

#             # Optionally print the error at each epoch
#             if epoch % 100 == 0:
#                 loss = np.mean(np.square(target_output - self.final_output))
#                 print(f'Epoch {epoch}, Loss: {loss:.4f}')

#     def predict(self, input_data):
#         probabilities = self.forward(input_data)
#         # Apply threshold to get binary classification
#         return (probabilities >= 0.5).astype(int)

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.output_layer = nn.Linear(prev, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))  # Apply sigmoid after each hidden layer
        x = torch.sigmoid(self.output_layer(x))  # Final output layer
        return x

    def train(self, input_data, target_output, epochs, learning_rate):
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(input_data)
            loss = criterion(output, target_output)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, input_data):
        with torch.no_grad():
            output = self.forward(input_data)
            return (output >= 0.5).int()
    
def encode_nn_in_z3(model, input_vars):
    """
    Encode a PyTorch FeedForwardNeuralNetwork with sigmoid activations (only),
    using a piecewise linear approximation of the sigmoid function.
    
    Returns:
        - constraints: list of Z3 constraints
        - output_vars: list of Z3 Real variables representing final outputs
    """
    constraints = []
    current_vars = input_vars

    def approx_sigmoid(x, name):
        """High-fidelity piecewise-linear sigmoid approximation for Z3."""
        y = Real(f"sigmoid_{name}")
        constraints = [
            If(x <= RealVal(-6), y == RealVal(0),
            If(x <= RealVal(-4), y == RealVal(0.05) * x + RealVal(0.3),
            If(x <= RealVal(-2), y == RealVal(0.1) * x + RealVal(0.5),
            If(x <= RealVal(0),  y == RealVal(0.2) * x + RealVal(0.5),
            If(x <= RealVal(2),  y == RealVal(0.1) * x + RealVal(0.5),
            If(x <= RealVal(4),  y == RealVal(0.05) * x + RealVal(0.7),
                                y == RealVal(1)))))))
        ]
        return y, constraints

    # Hidden layers
    for i, layer in enumerate(model.layers):
        weights = layer.weight.detach().numpy()
        biases = layer.bias.detach().numpy()
        layer_out = []

        for j in range(weights.shape[0]):
            weighted_sum = RealVal(biases[j])
            for k in range(len(current_vars)):
                weighted_sum += RealVal(weights[j][k]) * current_vars[k]

            sig_out, sig_constraints = approx_sigmoid(weighted_sum, f"{i}_{j}")
            constraints += sig_constraints
            layer_out.append(sig_out)

        current_vars = layer_out

    # Output layer
    output_weights = model.output_layer.weight.detach().numpy()
    output_biases = model.output_layer.bias.detach().numpy()
    output_vars = []

    for j in range(output_weights.shape[0]):
        weighted_sum = RealVal(output_biases[j])
        for k in range(len(current_vars)):
            weighted_sum += RealVal(output_weights[j][k]) * current_vars[k]

        sig_out, sig_constraints = approx_sigmoid(weighted_sum, f"output_{j}")
        constraints += sig_constraints
        output_vars.append(sig_out)

    return constraints, output_vars

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
        solver.add(x_vars[i] == float(x_sample[i].item()))
        
        # Perturbation bounds
        solver.add(x_perturbed[i] >= float(x_sample[i].item()) - epsilon)
        solver.add(x_perturbed[i] <= float(x_sample[i].item()) + epsilon)

    # diff = 1e-3
    # solver.add(Or([
    #     Abs(x_perturbed[i] - float(x_sample[i].item())) > diff
    #     for i in range(input_dim)
    # ]))
    

    # Encode the neural network for both original and perturbed inputs
    original_constraints, original_output_vars = encode_nn_in_z3(model, x_vars)
    perturbed_constraints, perturbed_output_vars = encode_nn_in_z3(model, x_perturbed)
    
    # Add all constraints to the solver
    for constraint in original_constraints:
        solver.add(constraint)
    
    for constraint in perturbed_constraints:
        solver.add(constraint)
    
    if expected_class == 1:
        solver.add(perturbed_output_vars[0] < 0.5)  # misclassified as 0
    else:
        solver.add(perturbed_output_vars[0] >= 0.5)  # misclassified as 1
    
    # Check satisfiability
    result = solver.check()
    
    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds")
    
    if result == sat:
        # Found a counterexample (adversarial example)
        model_solution = solver.model()
        
        # Extract the perturbed input values
        counterexample = torch.tensor([float(model_solution.eval(var).as_decimal(10).replace('?', '')) 
                                for var in x_perturbed])
        
        print(counterexample)
        
        # Find predicted class for perturbed input
        perturbed_outputs = []
        
        for j in range(num_classes):
            # Get the output variable's value from the model
            value = float(model_solution.eval(perturbed_output_vars[j]).as_decimal(10).replace('?', ''))
            perturbed_outputs.append(value)
        
        print("Predicted Probability ", perturbed_outputs[0])
        perturbed_class = int(perturbed_outputs[0] >= 0.5)
        print("Number of layers:", len(model.layers))

        print(f"Found adversarial example: Original class: {expected_class}, Perturbed class: {perturbed_class}")
        print(f"Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")
        
        return False, counterexample, perturbed_class
    else:
        print(f"No adversarial example found within epsilon = {epsilon}")
        return True, None, None

def adversarial_attack(model, x_sample, epsilon, expected_class, steps=100):
    """
    Perform a gradient-based adversarial attack for binary classifiers using BCELoss.
    
    Args:
        model: FeedForwardNeuralNetwork (PyTorch)
        x_sample: 1D NumPy array (original input)
        epsilon: maximum allowed perturbation (L-infinity norm)
        expected_class: original prediction class (0 or 1)
        steps: number of small steps to take

    Returns:
        adversarial_input: perturbed input that causes misclassification (if any)
        predicted_class: the predicted class for the perturbed input
    """
    # Prepare input tensor
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0)
    x_tensor.requires_grad = True

    # Define target (as float for BCELoss)
    target = torch.FloatTensor([[1 - expected_class]]) # DOUBLE CHECK 
    criterion = nn.BCELoss()

    for step in range(steps):
        output = model(x_tensor)
        loss = -criterion(output, target)  # maximize loss (negate it)
        loss.backward()

        # Get sign of gradient
        grad_sign = x_tensor.grad.sign().detach()
        
        # Apply perturbation step
        step_size = epsilon / steps
        x_tensor_adv = x_tensor.detach() + step_size * grad_sign
        x_tensor_adv = torch.clamp(x_tensor_adv, x_sample - epsilon, x_sample + epsilon)

        # Create new input and reset grad
        x_tensor = x_tensor_adv.clone().detach().requires_grad_(True)

        # Check classification result
        pred_prob = model(x_tensor).item()
        pred_class = int(pred_prob >= 0.5)

        if pred_class != expected_class:
            print(f"✅ Found adversarial example at step {step}")
            return x_tensor.squeeze().detach().numpy(), pred_class

    # If unsuccessful
    print("⚠️  No adversarial example found.")
    pred_prob = model(x_tensor).item()
    pred_class = int(pred_prob >= 0.5)
    return x_tensor.squeeze().detach().numpy(), pred_class

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
        print(f"  - Perturbation magnitude: {np.linalg.norm(counterexample_smt - x_sample.numpy())}")
        
        if perturbed_class_grad != expected_class:
            print("Gradient-based attack also found an adversarial example:")
            print(f"  - Perturbed class: {perturbed_class_grad}")
            print(f"  - Perturbation magnitude: {np.linalg.norm(x_perturbed_grad - x_sample.numpy())}")
        else:
            print("Gradient-based attack failed to find an adversarial example")
            print("This could indicate the SMT solver is more powerful for this case")
    
    return is_robust_smt, counterexample_smt, perturbed_class_smt, x_perturbed_grad, perturbed_class_grad

def save_model_to_files(model, params_file, architecture_file):
    """Save model parameters and architecture to files."""
    params = []
    for layer in model.layers:
        layer_params = {
            'weights': layer.weight.data.numpy(),
            'biases': layer.bias.data.numpy()
        }
        params.append(layer_params)

    # Include the output layer
    output_params = {
        'weights': model.output_layer.weight.data.numpy(),
        'biases': model.output_layer.bias.data.numpy()
    }
    params.append(output_params)

    np.save(params_file, np.array(params, dtype=object))

    # Save architecture correctly
    architecture = {
        'input_size': model.layers[0].weight.shape[1],
        'hidden_sizes': [layer.weight.shape[0] for layer in model.layers],
        'output_size': model.output_layer.weight.shape[0]
    }

    with open(architecture_file, 'w') as f:
        f.write(str(architecture))


def load_model_from_files(params_file, architecture_file):
    """Load a neural network model from parameter and architecture files."""
    with open(architecture_file, 'r') as f:
        arch = eval(f.read())

    input_size = arch['input_size']
    hidden_sizes = arch['hidden_sizes']
    output_size = arch['output_size']

    model = FeedForwardNeuralNetwork(input_size, hidden_sizes, output_size)

    params = np.load(params_file, allow_pickle=True)

    # Assign to hidden layers
    for i, layer in enumerate(model.layers):
        layer_params = params[i]
        layer.weight.data = torch.FloatTensor(layer_params['weights'])
        layer.bias.data = torch.FloatTensor(layer_params['biases'])

    # Assign to output layer
    output_params = params[-1]
    model.output_layer.weight.data = torch.FloatTensor(output_params['weights'])
    model.output_layer.bias.data = torch.FloatTensor(output_params['biases'])

    return model

def set_the_model(): 
    # Training - predict the success rate of someone based on the # of hours they study and the # of hours they sleep 
    input_data = np.array([
        [2, 9],  # 2 hours of study, 9 hours of sleep
        [1, 5],  # 1 hour of study, 5 hours of sleep
        [3, 6],  # 3 hours of study, 6 hours of sleep
        [4, 8],  # 4 hours of study, 8 hours of sleep
        [1, 10], # 1 hour of study, 10 hours of sleep - relatively bad training data (outliers)
        [4, 4], # 4 hours of study, 4 hours of sleep - relatively bad training data (outliers)
        [4, 10], # 4 hours of study, 10 hours of sleep  
        [3, 5], # 3 hours of study, 5 hours of sleep 
        [2, 8], # 2 hours of study, 8 hours of sleep - relatively bad training data (outliers) 
        [6, 3], # 6 hours of study, 3 hours of sleep - relatively bad training data (outliers)
        [5, 7], # 5 hours of study, 7 hours of sleep - relatively bad training data (outliers)
        [2,6],
    ])
    target_output = np.array([[1], [0], [1], [1], [0], [0], [1], [0], [0], [0], [0], [0]])  # Pass or fail (1 = pass, 0 = fail)

    # converting to tensors 
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    target_tensor = torch.tensor(target_output, dtype=torch.float32)

    # Normalize input data
    input_tensor = input_tensor / input_tensor.max(dim=0, keepdim=True).values

    # Initialize and train the network
    model = FeedForwardNeuralNetwork(input_size=2, hidden_sizes=[2], output_size=1)
    model.train(input_tensor, target_tensor, epochs=10000, learning_rate=0.1)

    # Test the network
    output = model.forward(input_tensor)
    print("Predicted Output:")
    print(output)

    # 3. Save the model to files
    params_file = 'nn_params.npy'
    architecture_file = 'nn_architecture.txt'
    save_model_to_files(model, params_file, architecture_file)
    
def main(): 
    # set_the_model()
    params_file = 'nn_params.npy'
    architecture_file = 'nn_architecture.txt'
    # 4. Load the model from files (in a real scenario, you'd just load it)
    print("Loading model from files...")
    model = load_model_from_files(params_file, architecture_file)

    # New input test [4, 6] ie. 4 hours of studying 6 hours of sleep 
    input = torch.tensor([3,9], dtype=torch.float32)
    # Save this during training
    feature_max = torch.tensor([6.0, 10.0])
    input = input / feature_max  # normalizing 

    output = model.forward(input)
    print("Predicted Output:")
    print(output)
    classified_output = model.predict(input)
    print("Classified Output (0 or 1):")
    print(classified_output)
    predicted_class = classified_output
    output_size = 1
    
    # Verify robustness with different epsilon values
    for epsilon in [0.1, 0.2, 0.3]:
        print(f"\n=== Robustness Verification (Epsilon = {epsilon}) ===")
        is_robust_smt, counterexample_smt, perturbed_class_smt, x_perturbed_grad, perturbed_class_grad = \
            compare_verification_methods(model, input, epsilon, predicted_class, output_size)
        
        if not is_robust_smt:
            print("\nFound adversarial example using SMT:")
            print(f"Original input: {input}")
            print(f"Perturbed input: {counterexample_smt}")
            print(f"Perturbation: {counterexample_smt - input}")
            print(f"Original class: {predicted_class}, Perturbed class: {perturbed_class_smt}")
    

if __name__ == "__main__":
    main()
  




    

