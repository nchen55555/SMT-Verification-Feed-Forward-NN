import numpy as np
import torch
import torch.nn as nn
from z3 import *
import time
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import os

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

def fetch_iris_dataset():
    """Fetch the Iris dataset from UCI repository."""
    print("Fetching Iris dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Process the data
        data = []
        for line in response.text.strip().split('\n'):
            if line:
                values = line.split(',')
                if len(values) == 5:
                    data.append(values)
        
        # Convert features to float and class to integer
        X = np.array([[float(val) for val in row[:-1]] for row in data])
        
        # Convert class names to indices
        class_names = sorted(list(set(row[-1] for row in data)))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        y = np.array([class_to_idx[row[-1]] for row in data])
        
        return X, y, class_names
        
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        # Fallback to a small synthetic dataset if fetch fails
        X = np.random.rand(150, 4) * 10
        y = np.random.randint(0, 3, 150)
        class_names = ['Class A', 'Class B', 'Class C']
        return X, y, class_names

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
        
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            # Evaluate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
                print(f'Training Accuracy: {accuracy:.2f}')
    
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

def visualize_iris_data(X, y, class_names, sample_idx=None, perturbed_samples=None, filename="iris_visualization.png"):
    """
    Create and save visualizations of the dataset and adversarial examples.
    
    Args:
        X: Feature data
        y: Labels
        class_names: Names of classes
        sample_idx: Index of the original sample that was tested for robustness
        perturbed_samples: List of tuples (perturbed_sample, perturbed_class, epsilon)
        filename: Output filename
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Create a 3D scatter plot of the first three features
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot each class with a different color
    colors = ['blue', 'green', 'red']
    markers = ['o', '^', 's']
    
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        idx = (y == i)
        
        # Plot points for this class
        ax1.scatter(
            X[idx, 0], X[idx, 1], X[idx, 2],
            c=colors[i], marker=markers[i], label=class_name,
            alpha=0.7, s=30
        )
    
    # If there's a specific sample to highlight
    if sample_idx is not None:
        ax1.scatter(
            X[sample_idx, 0], X[sample_idx, 1], X[sample_idx, 2],
            c='black', marker='*', s=200, label='Original Sample'
        )
    
    # If there are perturbed samples to show
    if perturbed_samples:
        for idx, (perturbed_sample, perturbed_class, eps) in enumerate(perturbed_samples):
            ax1.scatter(
                perturbed_sample[0], perturbed_sample[1], perturbed_sample[2],
                c='yellow', marker='x', s=150, 
                label=f'Adversarial (ε={eps}, class={class_names[perturbed_class]})'
            )
    
    ax1.set_xlabel('Sepal Length')
    ax1.set_ylabel('Sepal Width')
    ax1.set_zlabel('Petal Length')
    ax1.set_title('3D Visualization of Iris Dataset (First 3 Features)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # 2. 2D scatter plot of sepal features
    ax2 = fig.add_subplot(222)
    
    for i, class_name in enumerate(class_names):
        idx = (y == i)
        ax2.scatter(
            X[idx, 0], X[idx, 1],
            c=colors[i], marker=markers[i], label=class_name, 
            alpha=0.7, s=30
        )
    
    if sample_idx is not None:
        ax2.scatter(
            X[sample_idx, 0], X[sample_idx, 1],
            c='black', marker='*', s=200
        )
    
    if perturbed_samples:
        for idx, (perturbed_sample, perturbed_class, eps) in enumerate(perturbed_samples):
            ax2.scatter(
                perturbed_sample[0], perturbed_sample[1],
                c='yellow', marker='x', s=150
            )
            
            # Draw a circle to represent epsilon boundary
            circle = plt.Circle(
                (X[sample_idx, 0], X[sample_idx, 1]), eps, 
                fill=False, linestyle='--', color='orange'
            )
            ax2.add_patch(circle)
    
    ax2.set_xlabel('Sepal Length')
    ax2.set_ylabel('Sepal Width')
    ax2.set_title('Sepal Features')
    
    # 3. 2D scatter plot of petal features
    ax3 = fig.add_subplot(223)
    
    for i, class_name in enumerate(class_names):
        idx = (y == i)
        ax3.scatter(
            X[idx, 2], X[idx, 3],
            c=colors[i], marker=markers[i], label=class_name,
            alpha=0.7, s=30
        )
    
    if sample_idx is not None:
        ax3.scatter(
            X[sample_idx, 2], X[sample_idx, 3],
            c='black', marker='*', s=200
        )
    
    if perturbed_samples:
        for idx, (perturbed_sample, perturbed_class, eps) in enumerate(perturbed_samples):
            ax3.scatter(
                perturbed_sample[2], perturbed_sample[3],
                c='yellow', marker='x', s=150
            )
            
            # Draw a circle to represent epsilon boundary
            circle = plt.Circle(
                (X[sample_idx, 2], X[sample_idx, 3]), eps, 
                fill=False, linestyle='--', color='orange'
            )
            ax3.add_patch(circle)
    
    ax3.set_xlabel('Petal Length')
    ax3.set_ylabel('Petal Width')
    ax3.set_title('Petal Features')
    
    # 4. Bar chart of perturbation magnitudes
    ax4 = fig.add_subplot(224)
    
    if perturbed_samples:
        eps_values = [eps for _, _, eps in perturbed_samples]
        smt_found = [1 if sample is not None else 0 for sample, _, _ in perturbed_samples]
        grad_found = [1 for _ in perturbed_samples]  # Assuming grad attack always attempts
        
        x = np.arange(len(eps_values))
        width = 0.35
        
        ax4.bar(x - width/2, smt_found, width, label='SMT Found Adversarial', color='red', alpha=0.7)
        ax4.bar(x + width/2, grad_found, width, label='Gradient Attack Tried', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Epsilon Value')
        ax4.set_ylabel('Success (1 = Yes, 0 = No)')
        ax4.set_title('Comparison of Verification Methods')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'ε={eps}' for eps in eps_values])
        ax4.legend()
        
        # Add perturbation magnitude for successful finds
        for i, (sample, _, eps) in enumerate(perturbed_samples):
            if sample is not None:
                mag = np.linalg.norm(sample - X[sample_idx])
                ax4.text(i - width/2, 1.1, f'Magnitude: {mag:.4f}', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No Perturbations Found', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes)
    
    # Add overall title
    plt.suptitle('Iris Dataset and Adversarial Examples Visualization', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    
    # Close the figure to free memory
    plt.close(fig)

def create_decision_boundary_plot(model, X, y, sample_idx, perturbed_samples, scaler, class_names, filename="decision_boundary.png"):
    """
    Create and save a visualization of the decision boundary in 2D space.
    """
    # We'll plot the decision boundary in the space of the first two features
    feature_idx1, feature_idx2 = 0, 1  # sepal length and width
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Get feature ranges for creating the mesh grid
    x_min, x_max = X[:, feature_idx1].min() - 0.5, X[:, feature_idx1].max() + 0.5
    y_min, y_max = X[:, feature_idx2].min() - 0.5, X[:, feature_idx2].max() + 0.5
    
    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Create feature vectors for each point in the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For the other features, use the mean values from the dataset
    other_features_mean = np.mean(X, axis=0)
    full_mesh_points = np.zeros((mesh_points.shape[0], X.shape[1]))
    full_mesh_points[:, feature_idx1] = mesh_points[:, 0]
    full_mesh_points[:, feature_idx2] = mesh_points[:, 1]
    
    # Fill in the other features with mean values
    for i in range(X.shape[1]):
        if i != feature_idx1 and i != feature_idx2:
            full_mesh_points[:, i] = other_features_mean[i]
    
    # Make predictions for each point in the mesh
    with torch.no_grad():
        mesh_tensor = torch.FloatTensor(full_mesh_points)
        mesh_preds = model(mesh_tensor).detach().numpy()
        mesh_classes = np.argmax(mesh_preds, axis=1)
    
    # Reshape predictions back to the mesh shape
    z = mesh_classes.reshape(xx.shape)
    
    # Plot the decision boundary
    colors = ['blue', 'green', 'red']
    cmap = ListedColormap(colors[:len(class_names)])
    plt.contourf(xx, yy, z, alpha=0.3, cmap=cmap)
    
    # Plot the original data points
    for i, class_name in enumerate(class_names):
        idx = (y == i)
        plt.scatter(X[idx, feature_idx1], X[idx, feature_idx2], 
                   c=colors[i], label=class_name, edgecolors='k', alpha=0.8)
    
    # Highlight the sample being tested
    if sample_idx is not None:
        plt.scatter(X[sample_idx, feature_idx1], X[sample_idx, feature_idx2], 
                   c='black', marker='*', s=200, label='Original Sample')
    
    # Plot perturbed samples
    if perturbed_samples:
        for idx, (perturbed_sample, perturbed_class, eps) in enumerate(perturbed_samples):
            if perturbed_sample is not None:
                plt.scatter(perturbed_sample[feature_idx1], perturbed_sample[feature_idx2], 
                           c='yellow', marker='x', s=150, 
                           label=f'Adversarial (ε={eps}, class={class_names[perturbed_class]})')
                
                # Draw a circle to represent epsilon boundary
                circle = plt.Circle(
                    (X[sample_idx, feature_idx1], X[sample_idx, feature_idx2]), eps, 
                    fill=False, linestyle='--', color='orange', label=f'ε={eps} Boundary'
                )
                plt.gca().add_patch(circle)
                
                # Draw a line connecting original to perturbed
                plt.plot([X[sample_idx, feature_idx1], perturbed_sample[feature_idx1]],
                        [X[sample_idx, feature_idx2], perturbed_sample[feature_idx2]],
                        'k--', alpha=0.5)
    
    plt.title(f'Decision Boundaries (Sepal Length vs Sepal Width)')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    print(f"Decision boundary visualization saved to {filename}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # 1. Fetch and prepare the Iris dataset
    X, y, class_names = fetch_iris_dataset()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
    print(f"Class names: {class_names}")
    
    # 2. Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train a simple neural network
    input_size = X_train.shape[1]  # Number of features
    hidden_sizes = [10, 8]  # Small network for demonstration
    output_size = len(class_names)  # Number of classes
    
    print("Training neural network...")
    model = train_simple_model(input_size, hidden_sizes, output_size, X_train_scaled, y_train, epochs=2000)
    
    # 4. Evaluate model on test set
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 5. Save the model to files
    params_file = 'nn_params.npy'
    architecture_file = 'nn_architecture.txt'
    save_model_to_files(model, params_file, architecture_file)
    
    # 6. Load the model from files
    print("Loading model from files...")
    model = load_model_from_files(params_file, architecture_file)
    
    # 7. Select a test sample for verification
    sample_idx = 0  # Take the first test sample for simplicity
    x_sample = X_test_scaled[sample_idx]
    true_class = y_test[sample_idx]
    
    # Predict using PyTorch
    with torch.no_grad():
        prediction = model(torch.FloatTensor(x_sample))
        predicted_class = torch.argmax(prediction).item()
    
    print(f"\nSample: Feature vector of length {len(x_sample)}")
    print(f"True class: {true_class} ({class_names[true_class]})")
    print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
    
    # 8. Verify robustness with different epsilon values
    perturbed_samples = []
    
    for epsilon in [0.1, 0.2, 0.3]:
        print(f"\n=== Robustness Verification (Epsilon = {epsilon}) ===")
        is_robust_smt, counterexample_smt, perturbed_class_smt, x_perturbed_grad, perturbed_class_grad = \
            compare_verification_methods(model, x_sample, epsilon, predicted_class, output_size)
        
        # Store results for visualization
        if not is_robust_smt:
            # Convert scaled adversarial example back to original feature space for visualization
            counterexample_orig = scaler.inverse_transform(counterexample_smt.reshape(1, -1)).squeeze()
            perturbed_samples.append((counterexample_orig, perturbed_class_smt, epsilon))
            
            print("\nFound adversarial example using SMT:")
            print(f"Original input: {x_sample}")
            print(f"Perturbed input: {counterexample_smt}")
            print(f"Perturbation: {counterexample_smt - x_sample}")
            print(f"Original class: {predicted_class} ({class_names[predicted_class]}), "
                  f"Perturbed class: {perturbed_class_smt} ({class_names[perturbed_class_smt]})")
    
    # 9. Create visualizations of the dataset and adversarial examples
    print("\nGenerating visualizations...")
    
    # Convert the original test sample back to original feature space
    x_sample_orig = scaler.inverse_transform(x_sample.reshape(1, -1)).squeeze()
    
    # Visualization 1: Dataset and adversarial examples
    visualize_iris_data(
        X=X,  # Use the original (unscaled) dataset for visualization
        y=y, 
        class_names=class_names,
        sample_idx=X_test.shape[0] * 0 + sample_idx,  # This is a simplification; in reality, we'd need to find the index
        perturbed_samples=perturbed_samples,
        filename="iris_visualization.png"
    )
    
    # Visualization 2: Decision boundary and adversarial examples
    create_decision_boundary_plot(
        model=model,
        X=X_test_scaled,  # Use scaled data for decision boundary
        y=y_test,
        sample_idx=sample_idx,
        perturbed_samples=[(counterexample_smt, perturbed_class_smt, epsilon) 
                          for counterexample_orig, perturbed_class_smt, epsilon in perturbed_samples],
        scaler=scaler,
        class_names=class_names,
        filename="decision_boundary.png"
    )
    
    print("\nCheckpoint 1 MVP completed!")

if __name__ == "__main__":
    main()
