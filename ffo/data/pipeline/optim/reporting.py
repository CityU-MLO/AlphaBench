"""
Simple Factor Combination Backtesting Report

This module generates simple reports for factor combination optimization,
focusing on weights, performance metrics, and factor contributions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime


class SimpleFactorReporter:
    """
    Generate simple backtesting reports for factor combination results.

    Includes:
    - Learned factor weights for each model
    - Performance metrics
    - Individual factor contribution analysis
    """

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_backtest_report(
        self,
        model_results: Dict,
        backtest_performance: Optional[Dict] = None,
        report_title: str = "Factor Combination Backtest Report",
    ) -> Dict:
        """
        Generate simple backtest report.

        Args:
            model_results: Results from factor optimization (weights, performance)
            backtest_performance: Optional backtest performance metrics
            report_title: Title for the report

        Returns:
            Dict containing report data
        """

        report = {
            "title": report_title,
            "generated_at": datetime.now().isoformat(),
            "models": {},
            "summary": {},
        }

        # Process each model
        for model_name, results in model_results.items():
            model_report = self._generate_model_section(results)
            report["models"][model_name] = model_report

        # Add backtest performance if available
        if backtest_performance:
            report["backtest_performance"] = backtest_performance

        # Generate summary
        report["summary"] = self._generate_summary_section(report["models"])

        # Save to file
        self._save_report(report, report_title)

        return report

    def _generate_model_section(self, results: Dict) -> Dict:
        """Generate section for a single model."""

        section = {}

        # Factor weights
        if "weights" in results:
            weights = results["weights"]
            if isinstance(weights, pd.Series):
                section["factor_weights"] = weights.to_dict()
            else:
                section["factor_weights"] = weights

        # Performance metrics
        if "performance" in results:
            perf = results["performance"]
            section["performance_metrics"] = {
                "combined_ic_mean": perf.get("combined_ic_mean"),
                "combined_ic_ir": perf.get("combined_ic_ir"),
                "num_factors_used": perf.get("num_factors_used"),
                "method": perf.get("method"),
            }

        # Factor contributions (simplified)
        section["factor_contributions"] = self._analyze_factor_contributions(results)

        return section

    def _analyze_factor_contributions(self, results: Dict) -> Dict:
        """Analyze individual factor contributions."""

        contributions = {}

        if "weights" in results and "ic_df" in results:
            weights = results["weights"]
            ic_df = results["ic_df"]

            # Calculate weighted IC for each factor
            if isinstance(weights, pd.Series):
                weight_dict = weights.to_dict()
            else:
                weight_dict = weights

            for factor_name, weight in weight_dict.items():
                if factor_name in ic_df.columns and weight > 0:
                    factor_ic = ic_df[factor_name]
                    weighted_ic = factor_ic * weight

                    contributions[factor_name] = {
                        "weight": weight,
                        "avg_ic": factor_ic.mean(),
                        "weighted_contribution": weighted_ic.mean(),
                        "ic_volatility": factor_ic.std(),
                    }

        return contributions

    def _generate_summary_section(self, models: Dict) -> Dict:
        """Generate summary across all models."""

        summary = {"total_models": len(models), "model_comparison": {}}

        for model_name, model_data in models.items():
            perf = model_data.get("performance_metrics", {})
            summary["model_comparison"][model_name] = {
                "ic_mean": perf.get("combined_ic_mean"),
                "ic_ir": perf.get("combined_ic_ir"),
                "factors_used": perf.get("num_factors_used"),
            }

        return summary

    def _save_report(self, report: Dict, title: str) -> str:
        """Save report to JSON file."""

        import json

        # Create filename
        filename = f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Make serializable
        serializable_report = self._make_serializable(report)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        print(f"Report saved to: {filepath}")
        return filepath

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return obj


def print_report_summary(report: Dict):
    """Print a human-readable summary of the report."""

    print(f"\n{'='*60}")
    print(f"📊 {report['title']}")
    print(f"{'='*60}")
    print(f"Generated: {report['generated_at'][:19]}")
    print(f"Models analyzed: {report['summary']['total_models']}")
    print()

    # Model details
    for model_name, model_data in report["models"].items():
        print(f"🔹 {model_name.upper()}")
        print("-" * 30)

        # Performance
        perf = model_data.get("performance_metrics", {})
        if perf:
            ic_mean = perf.get("combined_ic_mean")
            ic_ir = perf.get("combined_ic_ir")
            factors_used = perf.get("num_factors_used")
            method = perf.get("method")

            if ic_mean is not None:
                print(f"  IC Mean: {ic_mean:.4f}")
            if ic_ir is not None:
                print(f"  IC IR: {ic_ir:.4f}")
            if factors_used is not None:
                print(f"  Factors Used: {factors_used}")
            if method:
                print(f"  Method: {method}")

            # LASSO-specific metrics
            r2 = perf.get("final_r2")
            sparsity = perf.get("sparsity_ratio")
            if r2 is not None:
                print(f"  R² Score: {r2:.4f}")
            if sparsity is not None:
                print(f"  Sparsity: {sparsity:.2%}")

        # Weights
        weights = model_data.get("factor_weights", {})
        if weights:
            print("  Factor Weights:")
            for factor, weight in weights.items():
                if weight > 0:
                    print(f"    {factor}: {weight:.4f}")

        # Top contributions
        contributions = model_data.get("factor_contributions", {})
        if contributions:
            print("  Top Factor Contributions:")
            sorted_contribs = sorted(
                contributions.items(),
                key=lambda x: x[1]["weighted_contribution"],
                reverse=True,
            )
            for factor, data in sorted_contribs[:3]:
                print(
                    f"    {factor}: {data['weighted_contribution']:.4f} (weight: {data['weight']:.4f})"
                )

        print()

    # Backtest performance
    if "backtest_performance" in report:
        print("📈 Backtest Performance")
        print("-" * 30)
        backtest = report["backtest_performance"]
        # Add backtest metrics here if available
        print("(Backtest metrics would be displayed here)")
        print()


def generate_sample_report():
    """Generate a sample report for demonstration."""

    # Mock IC optimization results
    mock_ic_results = {
        "weights": {
            "Div($close, Mean($close, 5))": 0.4,
            "Sub($close, Ref($close, 1))": 0.3,
            "Mean($volume, 10)": 0.3,
        },
        "performance": {
            "combined_ic_mean": 0.025,
            "combined_ic_ir": 0.35,
            "num_factors_used": 3,
            "method": "equal_top",
        },
        "ic_df": pd.DataFrame(
            {
                "Div($close, Mean($close, 5))": np.random.normal(0.02, 0.05, 100),
                "Sub($close, Ref($close, 1))": np.random.normal(0.015, 0.04, 100),
                "Mean($volume, 10)": np.random.normal(0.01, 0.03, 100),
            }
        ),
    }

    # Mock LASSO results
    mock_lasso_results = {
        "weights": {"factor_1": 0.6, "factor_2": 0.0, "factor_3": 0.4, "factor_4": 0.0},
        "performance": {"final_r2": 0.12, "sparsity_ratio": 0.5, "num_factors_used": 2},
    }

    model_results = {
        "ic_optimization": mock_ic_results,
        "lasso_regression": mock_lasso_results,
    }

    # Generate report
    reporter = SimpleFactorReporter()
    report = reporter.generate_backtest_report(
        model_results=model_results, report_title="Sample Factor Combination Report"
    )

    # Print summary
    print_report_summary(report)

    return report


if __name__ == "__main__":
    try:
        report = generate_sample_report()
    except Exception as e:
        print(f"Report generation failed: {e}")
        import traceback

        traceback.print_exc()
