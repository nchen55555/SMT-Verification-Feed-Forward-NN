# Feed Forward Neural Network Verification 

Our project investigates the use of Satisfiability Modulo Theories (SMT) solvers for verifying the robustness of neural networks against adversarial perturbations. We will take a trained neural network classifier that we already have the two files for (parameters and architecture) and convert it into a system of constraints for an SMT solver. In Checkpoint 1, we indicated that by Checkpoint 2, we wanted to:
Extract and encode the neural network’s parameters and structure into a format suitable for an SMT solver
Construct the corresponding constraints within the solver
Conduct an MVP robustness evaluation
We will measure success by determining whether the SMT solver correctly identifies perturbations that can impact the classification result. 

