"""
Statistical analysis for benchmarking results.
"""

from typing import Any, Dict, List

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis for method comparison."""

    def analyze_method_comparison(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on method comparison."""
        methods = list(aggregated_results.keys())

        if len(methods) < 2:
            return {"error": "Need at least 2 methods for comparison"}

        analysis = {
            "method_rankings": self._rank_methods(aggregated_results),
            "pairwise_comparisons": self._pairwise_comparisons(aggregated_results),
            "effect_sizes": self._calculate_effect_sizes(aggregated_results),
            "summary": self._generate_summary(aggregated_results),
        }

        return analysis

    def _rank_methods(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank methods by performance."""
        rankings = []

        for method, data in results.items():
            rankings.append(
                {
                    "method": method,
                    "mean_performance": data["mean_performance"],
                    "std_performance": data["std_performance"],
                    "success_rate": data["success_rate"],
                }
            )

        # Sort by mean performance
        rankings.sort(key=lambda x: x["mean_performance"], reverse=True)

        # Add ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        return rankings

    def _pairwise_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pairwise statistical comparisons."""
        methods = list(results.keys())
        comparisons = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                data1 = results[method1]["raw_data"]["final_performances"]
                data2 = results[method2]["raw_data"]["final_performances"]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)

                comparison_key = f"{method1}_vs_{method2}"
                comparisons[comparison_key] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "winner": method1 if np.mean(data1) > np.mean(data2) else method2,
                    "effect_size": self._cohen_d(data1, data2),
                }

        return comparisons

    def _cohen_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _calculate_effect_sizes(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effect sizes for each method."""
        effect_sizes = {}

        for method, data in results.items():
            improvements = data["raw_data"]["improvements"]
            if improvements:
                # Effect size as improvement over baseline
                effect_sizes[method] = np.mean(improvements) / np.std(improvements) if np.std(improvements) > 0 else 0.0

        return effect_sizes

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        best_method = max(results.keys(), key=lambda m: results[m]["mean_performance"])

        return {
            "best_method": best_method,
            "best_performance": results[best_method]["mean_performance"],
            "total_methods": len(results),
            "total_runs": sum(data["total_runs"] for data in results.values()),
        }
