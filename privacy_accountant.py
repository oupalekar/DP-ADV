import torch
class PrivacyAccountant():
    def __init__(self, noise, q) -> None:
        """ 
        Here we compare them using some concrete values.
        The overall privacy loss (ε,δ) can be computed from the noise level σ,
        the sampling ra- tio of each lot q = L/N (so each epoch consists of 1/q batches),
        and the number of epochs E (so the number of steps is T = E/q). We fix the target δ = 10−5,
        the value used for our MNIST and CIFAR experiments."""
        self.noise = noise
        self.q = q
        self.steps = 0

    def compute_log_moment(self, lambda_):
        log_moment = (self.q**2) * (lambda_ * (lambda_ + 1)) / (4 * self.noise**2)

        return log_moment * self.steps
    
    def get_privacy_budget(self, target_delta):
        lambda_range = range(1, 100)

        epsilon = float('inf')

        for lambda_ in lambda_range:
            log_moment = self.compute_log_moment(lambda_)

            possible_epsilon = (log_moment + torch.log(torch.tensor(1.0/target_delta))) / lambda_
            epsilon =  min(epsilon, possible_epsilon.item())

        return epsilon
    
    def step(self, num_steps = 1):
        self.steps += num_steps