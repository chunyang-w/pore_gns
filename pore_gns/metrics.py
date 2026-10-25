def dice(a, b):
    """2|a & b| / (|a| + |b|) of two binary fields."""
    a, b = a.float(), b.float()
    return float(2 * (a * b).sum() / (a.sum() + b.sum()))
