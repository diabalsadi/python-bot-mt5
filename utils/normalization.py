import numpy as np

def safe_clip(value, lower=-5.0, upper=5.0):
    return float(np.clip(value, lower, upper))

def robust_normalize(value, mean, std):
    if std is None or std == 0:
        return 0.0

    normalized = (value - mean) / std
    return safe_clip(normalized)
