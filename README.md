# Formalizing Neural Network Robustness against Adversarial Perturbations  

We investigates the use of Satisfiability Modulo Theories (SMT) solvers for verifying the robustness of neural networks against adversarial perturbations. We will take a trained neural network classifier or train one ourselves that we already have the two files for (parameters and architecture) and convert it into a system of constraints for an SMT solver. 


We use an SMT-solver (ie. Z3) to verify the robustness of simple feed-forward neural networks against adversarial perturbations of epsilon. 
Accordingly, we analyze two security guarantees in this investigation
- Adversarial robustness - small perturbations of an input do not change the classification
- Stability (side quest) - small changes in input do not cause large changes in the output

## Motivation 
- We believe SMT Solvers could give stronger, more reliable robustness certificates based on existing literature
- We want to determine whether this framework can be validated across different architectures and tasks

## Methodology and Approach 
1. We both train our own models as well as call existing models loaded on HuggingFace
2. We then convert the linear weights and activation functions into constraints in our SMT-solver
3. We specify an original input vector x that and a perturbed input vector x_perturbed, adding that the class predicted by x != the class predicted by x_perturbed
4. If the SMT-solver outputs SAT then our model is not robust by our SMT-solver. UNSAT means our model is considered robust
5. We subsequently run a gradient-based adversarial attack on our model to see if this traditionally-used method finds an adversarial attack
6. Compare the two methodologies to see if they agree/disagree, measuring their overlap as a success metric

## Measuring Success 
Measure success via the similarity between the set of models noted robust by the SMT-solver and the set noted by Gradient Descent

We want to compare the two methods to see what works best! 



