import numpy as np

def compute_uncertainty_metrics(ensemble_preds, confidence=0.90):
    """
    Computes mean prediction, standard deviation, and confidence intervals.
    """
    mean_pred = np.mean(ensemble_preds, axis=0)
    std_dev = np.std(ensemble_preds, axis=0)

    # Compute Confidence Interval (assuming normal distribution)
    ci_multiplier = 1.645  # For 90% confidence interval
    ci_lower = mean_pred - ci_multiplier * std_dev
    ci_upper = mean_pred + ci_multiplier * std_dev

    return {
        "mean_pred": mean_pred,
        "std_dev": std_dev,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }
