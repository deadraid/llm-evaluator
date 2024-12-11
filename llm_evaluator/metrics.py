from typing import Dict, List

import numpy as np


class MetricsCalculator:
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config

    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate all metrics from results"""
        metrics = {}
        total = results.get('total', 0)

        if total == 0:
            # Return default metrics with 0 values
            base_metrics = {f"pass@{k}": 0.0 for k in self.metrics_config.k_values}
            base_metrics.update({
                "average_pass@k": 0.0,
                "weighted_pass@k": 0.0 if self.metrics_config.weighted_k else None
            })
            return base_metrics

        # Calculate Pass@K metrics
        pass_k_metrics = {}
        for k in self.metrics_config.k_values:
            metric_key = f"pass@{k}"
            pass_k_metrics[metric_key] = (results['metrics'].get(metric_key, 0) / total) * 100

        # Calculate average
        avg_pass_k = sum(pass_k_metrics.values()) / len(pass_k_metrics)
        metrics = {
            **pass_k_metrics,
            "average_pass@k": avg_pass_k
        }

        # Calculate weighted average if configured
        if self.metrics_config.weighted_k and self.metrics_config.weights:
            weighted_sum = sum(
                metrics[f"pass@{k}"] * self.metrics_config.weights[k]
                for k in self.metrics_config.k_values
            )
            total_weight = sum(self.metrics_config.weights.values())
            metrics["weighted_pass@k"] = weighted_sum / total_weight if total_weight > 0 else 0.0

        return metrics

    def calculate_speed_metrics(self, timestamps: List[float]) -> Dict:
        """Calculate speed-related metrics"""
        if len(timestamps) <= 1:
            return {"avg_time": 0.0, "std_time": 0.0}

        times = np.diff(timestamps)
        return {
            "avg_time": float(np.mean(times)),
            "std_time": float(np.std(times))
        }
