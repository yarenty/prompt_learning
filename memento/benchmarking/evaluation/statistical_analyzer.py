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

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate summary of statistical analysis."""
        best_method = max(results.keys(), key=lambda k: results[k]["mean_performance"])
        return f"Best performing method: {best_method} with mean performance {results[best_method]['mean_performance']:.3f}"

    def calculate_descriptive_stats(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate descriptive statistics for performance data."""
        if not performance_data:
            return {"error": "No performance data provided"}

        # Extract performance scores
        scores = []
        domains = set()

        for data_point in performance_data:
            if "performance" in data_point:
                scores.append(data_point["performance"])
            if "domain" in data_point:
                domains.add(data_point["domain"])

        if not scores:
            return {"error": "No performance scores found in data"}

        scores_array = np.array(scores)

        return {
            "count": len(scores),
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75)),
            "domains_covered": list(domains),
            "domain_count": len(domains),
        }

    def analyze_by_domain(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by domain."""
        domain_stats = {}

        for data_point in performance_data:
            domain = data_point.get("domain", "unknown")
            performance = data_point.get("performance", 0)

            if domain not in domain_stats:
                domain_stats[domain] = []
            domain_stats[domain].append(performance)

        # Calculate stats for each domain
        domain_analysis = {}
        for domain, scores in domain_stats.items():
            scores_array = np.array(scores)
            domain_analysis[domain] = {
                "count": len(scores),
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array)),
            }

        return domain_analysis

    def calculate_confidence_intervals(
        self, performance_data: List[Dict[str, Any]], confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for performance data."""
        scores = [d.get("performance", 0) for d in performance_data if "performance" in d]

        if len(scores) < 2:
            return {"error": "Need at least 2 data points for confidence intervals"}

        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        sem = stats.sem(scores_array)  # Standard error of the mean

        # Calculate confidence interval
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, len(scores) - 1)
        margin_of_error = t_critical * sem

        return {
            "mean": float(mean),
            "confidence_level": confidence,
            "lower_bound": float(mean - margin_of_error),
            "upper_bound": float(mean + margin_of_error),
            "margin_of_error": float(margin_of_error),
            "sample_size": len(scores),
        }

    def calculate_effect_sizes(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate effect sizes (Cohen's d) for method comparisons."""
        # Group data by method if available
        method_scores = {}

        for data_point in performance_data:
            method = data_point.get("method", "unknown")
            performance = data_point.get("performance", 0)

            if method not in method_scores:
                method_scores[method] = []
            method_scores[method].append(performance)

        if len(method_scores) < 2:
            return {"error": "Need at least 2 methods for effect size calculation"}

        methods = list(method_scores.keys())
        effect_sizes = {}

        # Calculate pairwise effect sizes
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                scores1 = np.array(method_scores[method1])
                scores2 = np.array(method_scores[method2])

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(scores1) - 1) * np.var(scores1, ddof=1) + (len(scores2) - 1) * np.var(scores2, ddof=1))
                    / (len(scores1) + len(scores2) - 2)
                )

                if pooled_std > 0:
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    effect_sizes[f"{method1} vs {method2}"] = float(cohens_d)

        return effect_sizes
