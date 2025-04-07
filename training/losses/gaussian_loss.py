import torch
import torch.nn as nn

class GaussianLoss(nn.Module):
    """
    Implements Gaussian Negative Log-Likelihood (NLL) loss for probabilistic regression.

    The model must predict both:
      - `mean`: Expected value (first output channel)
      - `variance`: Predicted log-variance (log_sigma^2)
    
    This loss penalizes overconfident incorrect predictions and models input-dependent (heteroscedastic) uncertainty.
    """

    def __init__(self, ignore_value=None, pre_calculation_function=None, epsilon=1e-8):
        """
        :param ignore_value: Value in target to ignore during loss calculation.
        :param pre_calculation_function: Function to preprocess input-output pairs.
        :param epsilon: Small constant to prevent log(0) issues.
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.epsilon = epsilon  # Ensures variance is positive

    def forward(self, out, target):
        """
        Compute Gaussian Negative Log-Likelihood Loss.

        :param out: Tensor of shape (batch, 2, H, W)
                    - First channel: Mean prediction
                    - Second channel: Log-variance prediction
        :param target: Ground truth tensor (batch, 1, H, W)
        :return: Scalar loss value
        """
        if self.pre_calculation_function is not None:
            out, target = self.pre_calculation_function(out, target)

        mean = out[:, 0]  # Extract mean
        log_variance = out[:, 1]  # Extract variance

        # Ensure variance is positive to avoid log(0)
        # variance = torch.abs(variance) + self.epsilon  

        # Flatten tensors to match L2Loss implementation
        mean = mean.flatten()
        log_variance = log_variance.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            mask = target != self.ignore_value
            mean = mean[mask]
            log_variance = log_variance[mask]
            target = target[mask]

        # Convert log_variance to variance
        variance = torch.exp(log_variance) + self.epsilon 

        # Compute Gaussian NLL loss
        # nll_loss = ((target - mean) ** 2) / (2 * variance) + 0.5 * torch.log(variance)
        nll_loss = 0.5 * ((target - mean) ** 2 / variance + log_variance)

        return torch.mean(nll_loss) #nll_loss.mean()  # Return mean loss across batch
