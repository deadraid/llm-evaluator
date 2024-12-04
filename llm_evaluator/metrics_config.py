import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics"""
    k_values: List[int]  # List of K values for Pass@K
    weighted_k: bool     # Whether to use weighted Pass@K
    weights: Optional[Dict[int, float]] = None  # Weights for different K positions

def parse_k_values() -> List[int]:
    """Parse K values from environment variable"""
    k_values_str = os.getenv("PASS_K_VALUES", "1")
    try:
        return sorted([int(k.strip()) for k in k_values_str.split(",")])
    except ValueError as e:
        raise ValueError(f"Invalid K values format. Use comma-separated integers. Error: {e}")

def parse_k_weights() -> Optional[Dict[int, float]]:
    """Parse weights for different K positions from environment variable"""
    weights_str = os.getenv("PASS_K_WEIGHTS", None)
    if not weights_str:
        return None

    try:
        # Format should be "k1:w1,k2:w2,..." e.g., "1:1.0,2:0.8,3:0.6"
        weights = {}
        pairs = weights_str.split(",")
        for pair in pairs:
            k, w = pair.split(":")
            weights[int(k.strip())] = float(w.strip())
        return weights
    except ValueError as e:
        raise ValueError(f"Invalid weights format. Use 'k1:w1,k2:w2,...'. Error: {e}")

def get_metrics_config() -> MetricsConfig:
    """Get metrics configuration from environment variables"""
    k_values = parse_k_values()
    use_weighted = os.getenv("USE_WEIGHTED_PASS_K", "false").lower() == "true"
    weights = parse_k_weights() if use_weighted else None

    # Validate weights if using weighted Pass@K
    if use_weighted and weights:
        # Check if we have weights for all K values
        missing_weights = [k for k in k_values if k not in weights]
        if missing_weights:
            raise ValueError(f"Missing weights for K values: {missing_weights}")

    return MetricsConfig(
        k_values=k_values,
        weighted_k=use_weighted,
        weights=weights
    )
