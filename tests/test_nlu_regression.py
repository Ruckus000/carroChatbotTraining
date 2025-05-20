#!/usr/bin/env python3
"""
NLU Model Regression Test - Detects performance degradation between model versions
with configurable thresholds, statistical significance testing, and CI integration.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nlu_regression')

# Type aliases for clarity
MetricsDict = Dict[str, Any]
RegressionResult = Dict[str, Any]

class RegressionTester:
    """
    Smart regression testing for NLU models with statistical validation
    and CI-friendly output.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        'thresholds': {
            'intent_f1': 0.01,          # 1% decrease in intent F1
            'entity_f1': 0.02,          # 2% decrease in entity F1
            'accuracy': 0.01,           # 1% decrease in accuracy
            'high_impact_intents': 0.03  # 3% decrease for critical intents
        },
        'high_impact_intents': [],       # List of business-critical intents
        'significance_level': 0.05,      # p-value threshold for statistical significance
        'min_samples': 5,                # Minimum samples needed for statistical testing
        'metrics_to_track': ['intent_f1', 'entity_f1', 'accuracy'],
        'ignore_metrics': []             # Metrics to ignore in regression testing
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the regression tester with optional custom configuration.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self.DEFAULT_CONFIG.copy()

        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config and isinstance(custom_config, dict):
                        # Update only provided values, keep defaults for others
                        self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom configuration: {str(e)}")

        # Setup history tracking
        self.history_file = None
        self.history_df = None

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary recursively, preserving existing values"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def load_history(self, history_path: str) -> bool:
        """
        Load metrics history from CSV file.

        Args:
            history_path: Path to metrics history CSV

        Returns:
            bool: True if loaded successfully
        """
        self.history_file = history_path

        if not os.path.exists(history_path):
            logger.warning(f"History file not found: {history_path}")
            return False

        try:
            self.history_df = pd.read_csv(history_path)
            logger.info(f"Loaded history data with {len(self.history_df)} entries")
            return True
        except Exception as e:
            logger.error(f"Failed to load history file: {str(e)}")
            return False

    def load_metrics(self, metrics_path: str) -> Optional[MetricsDict]:
        """
        Load metrics from a JSON file.

        Args:
            metrics_path: Path to metrics JSON file

        Returns:
            MetricsDict or None if loading failed
        """
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Loaded metrics from {metrics_path}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics file {metrics_path}: {str(e)}")
            return None

    def _extract_key_metrics(self, metrics: MetricsDict) -> Dict[str, float]:
        """
        Extract key metrics from the full metrics dictionary.

        Args:
            metrics: Full metrics dictionary

        Returns:
            Dict[str, float]: Dictionary of key metrics
        """
        result = {}

        # Extract intent metrics
        intent_metrics = metrics.get('intent_metrics', {})
        result['intent_f1'] = intent_metrics.get('f1', 0.0)
        result['accuracy'] = intent_metrics.get('accuracy', 0.0)
        result['intent_precision'] = intent_metrics.get('precision', 0.0)
        result['intent_recall'] = intent_metrics.get('recall', 0.0)

        # Extract per-class intent metrics for high impact intents
        per_class = intent_metrics.get('per_class_report', {})
        for intent in self.config['high_impact_intents']:
            if intent in per_class:
                result[f'intent_{intent}_f1'] = per_class[intent].get('f1-score', 0.0)

        # Extract entity metrics
        entity_metrics = metrics.get('entity_metrics', {})
        micro_avg = entity_metrics.get('micro avg', {})
        result['entity_f1'] = micro_avg.get('f1-score', 0.0)
        result['entity_precision'] = micro_avg.get('precision', 0.0)
        result['entity_recall'] = micro_avg.get('recall', 0.0)

        # Extract error analysis
        error_analysis = metrics.get('error_analysis', {})
        result['intent_error_rate'] = error_analysis.get('intent_error_rate', 0.0)
        result['entity_error_rate'] = error_analysis.get('entity_error_rate', 0.0)

        return result

    def _get_best_previous_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics from previous runs.

        Returns:
            Dict[str, float]: Dictionary of best metrics
        """
        if self.history_df is None or len(self.history_df) == 0:
            return {}

        best_metrics = {}

        # For each metric, find the best value
        for metric in self.config['metrics_to_track']:
            if metric in self.history_df.columns:
                if metric.endswith('error_rate'):
                    # For error rates, lower is better
                    best_value = self.history_df[metric].min()
                else:
                    # For all other metrics, higher is better
                    best_value = self.history_df[metric].max()
                best_metrics[metric] = best_value

        # For high impact intents
        for intent in self.config['high_impact_intents']:
            metric = f'intent_{intent}_f1'
            if metric in self.history_df.columns:
                best_metrics[metric] = self.history_df[metric].max()

        return best_metrics

    def _check_statistical_significance(
        self,
        metric_name: str,
        current_value: float,
        regression_amount: float
    ) -> bool:
        """
        Check if the regression is statistically significant.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            regression_amount: Amount of regression

        Returns:
            bool: True if the regression is statistically significant
        """
        if self.history_df is None or len(self.history_df) < self.config['min_samples']:
            # Not enough data for statistical testing
            return True

        if metric_name not in self.history_df.columns:
            return True

        # Get historical values for this metric
        historical_values = self.history_df[metric_name].dropna().values

        if len(historical_values) < self.config['min_samples']:
            return True

        # Perform one-sample t-test
        t_stat, p_value = stats.ttest_1samp(historical_values, current_value)

        # Check if regression is significant
        if p_value < self.config['significance_level'] and t_stat > 0:
            return True

        return False

    def _get_regression_detail(
        self,
        metric_name: str,
        current_value: float,
        best_value: float,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Get detailed information about a regression.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            best_value: Best historical value
            threshold: Regression threshold

        Returns:
            Dict[str, Any]: Regression details
        """
        regression_amount = best_value - current_value
        is_significant = self._check_statistical_significance(
            metric_name, current_value, regression_amount
        )

        return {
            'metric': metric_name,
            'current_value': current_value,
            'best_value': best_value,
            'regression_amount': regression_amount,
            'regression_percent': (regression_amount / best_value) * 100 if best_value > 0 else 0,
            'threshold': threshold,
            'is_significant': is_significant,
            'is_regression': regression_amount > threshold and is_significant
        }

    def check_for_regression(
        self,
        current_metrics: MetricsDict
    ) -> Tuple[bool, RegressionResult]:
        """
        Check if the current metrics show a regression compared to historical bests.

        Args:
            current_metrics: Current metrics dictionary

        Returns:
            Tuple[bool, Dict]: (has_regressed, regression_details)
        """
        # Extract key metrics from full metrics
        current_key_metrics = self._extract_key_metrics(current_metrics)

        # Get best metrics from history
        best_metrics = self._get_best_previous_metrics()

        # If no history, this is the baseline
        if not best_metrics:
            logger.info("No history found. This will be treated as the baseline.")
            return False, {
                'has_regressed': False,
                'regressions': [],
                'current_metrics': current_key_metrics,
                'best_metrics': {}
            }

        # Check each metric for regression
        regressions = []

        for metric_name, current_value in current_key_metrics.items():
            # Skip metrics in ignore list
            if metric_name in self.config['ignore_metrics']:
                continue

            # Only check metrics we're tracking
            if (metric_name not in self.config['metrics_to_track'] and
                not any(metric_name.startswith(f'intent_{intent}_f1')
                       for intent in self.config['high_impact_intents'])):
                continue

            # Get best value and threshold
            best_value = best_metrics.get(metric_name, 0.0)

            # Determine threshold based on metric type
            if any(metric_name.startswith(f'intent_{intent}_f1')
                  for intent in self.config['high_impact_intents']):
                threshold = self.config['thresholds']['high_impact_intents']
            elif metric_name in self.config['thresholds']:
                threshold = self.config['thresholds'][metric_name]
            else:
                # Default to intent_f1 threshold
                threshold = self.config['thresholds']['intent_f1']

            # Check for regression
            if current_value < best_value - threshold:
                regression_detail = self._get_regression_detail(
                    metric_name, current_value, best_value, threshold
                )

                if regression_detail['is_regression']:
                    regressions.append(regression_detail)

        # Determine if there's a regression overall
        has_regressed = len(regressions) > 0

        return has_regressed, {
            'has_regressed': has_regressed,
            'regressions': regressions,
            'current_metrics': current_key_metrics,
            'best_metrics': best_metrics
        }

    def generate_report(
        self,
        regression_result: RegressionResult,
        output_format: str = 'text'
    ) -> str:
        """
        Generate a human-readable report of regression results.

        Args:
            regression_result: Result from check_for_regression
            output_format: Format of the report ('text', 'json', 'github')

        Returns:
            str: Formatted report
        """
        if output_format == 'json':
            return json.dumps(regression_result, indent=2)

        if output_format == 'github':
            return self._generate_github_report(regression_result)

        # Default to text report
        report = []

        if regression_result['has_regressed']:
            report.append("🚨 PERFORMANCE REGRESSION DETECTED 🚨\n")

            # Add details for each regression
            for i, reg in enumerate(regression_result['regressions']):
                report.append(f"Regression {i+1}: {reg['metric']}")
                report.append(f"  Current value: {reg['current_value']:.4f}")
                report.append(f"  Previous best: {reg['best_value']:.4f}")
                report.append(f"  Decrease: {reg['regression_amount']:.4f} ({reg['regression_percent']:.2f}%)")
                report.append(f"  Threshold: {reg['threshold']:.4f}")
                report.append(f"  Statistically significant: {'Yes' if reg['is_significant'] else 'No'}")
                report.append("")
        else:
            report.append("✅ No significant performance regression detected.\n")

        # Add summary of current metrics
        report.append("Current Metrics:")
        for metric, value in sorted(regression_result['current_metrics'].items()):
            if metric in regression_result['best_metrics']:
                best = regression_result['best_metrics'][metric]
                diff = value - best
                diff_str = f" ({'↑' if diff >= 0 else '↓'}{abs(diff):.4f})"
            else:
                diff_str = " (baseline)"

            report.append(f"  {metric}: {value:.4f}{diff_str}")

        return "\n".join(report)

    def _generate_github_report(self, regression_result: RegressionResult) -> str:
        """Generate a GitHub-compatible Markdown report for GitHub Actions"""
        report = []

        if regression_result['has_regressed']:
            report.append("## 🚨 Performance Regression Detected\n")

            # Add table for regressions
            report.append("| Metric | Current | Previous Best | Decrease | Threshold |")
            report.append("| ------ | -------:| -------------:| --------:| ---------:|")

            for reg in regression_result['regressions']:
                report.append(
                    f"| {reg['metric']} | {reg['current_value']:.4f} | {reg['best_value']:.4f} | "
                    f"{reg['regression_amount']:.4f} ({reg['regression_percent']:.2f}%) | {reg['threshold']:.4f} |"
                )

            report.append("\n### Recommendations")
            report.append("- Review recent changes to training data or model architecture")
            report.append("- Check for data drift or class imbalance issues")
            report.append("- Validate that benchmark data is representative")
        else:
            report.append("## ✅ No Performance Regression\n")
            report.append("All metrics are within acceptable thresholds compared to previous best results.")

        # Add summary of current metrics
        report.append("\n### Current Metrics\n")
        report.append("| Metric | Current | Previous Best | Difference |")
        report.append("| ------ | -------:| -------------:| ----------:|")

        for metric, value in sorted(regression_result['current_metrics'].items()):
            if metric in regression_result['best_metrics']:
                best = regression_result['best_metrics'][metric]
                diff = value - best
                diff_str = f"{'↑' if diff >= 0 else '↓'}{abs(diff):.4f} ({abs(diff/best*100):.2f}%)"
            else:
                diff_str = "baseline"

            report.append(f"| {metric} | {value:.4f} | {best:.4f if metric in regression_result['best_metrics'] else 'N/A'} | {diff_str} |")

        return "\n".join(report)

def main():
    """Main entry point for regression testing"""
    parser = argparse.ArgumentParser(description='Check for NLU model performance regression')
    parser.add_argument('--benchmark', default='data/nlu_benchmark_data.json',
                      help='Path to benchmark data')
    parser.add_argument('--model', default='trained_nlu_model',
                      help='Path to trained model')
    parser.add_argument('--output-dir', default='benchmark_results',
                      help='Output directory for results')
    parser.add_argument('--metrics-file',
                      help='Path to metrics file (if already generated)')
    parser.add_argument('--history-file',
                      help='Path to metrics history file')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--format', choices=['text', 'json', 'github'], default='text',
                      help='Output format for regression report')
    parser.add_argument('--ci', action='store_true',
                      help='Run in CI mode (exit with error code if regression detected)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize regression tester
    tester = RegressionTester(args.config)

    # Load metrics history
    history_file = args.history_file or os.path.join(args.output_dir, "metrics_history.csv")
    tester.load_history(history_file)

    # Get current metrics (either from file or by running evaluation)
    current_metrics = None

    if args.metrics_file and os.path.exists(args.metrics_file):
        # Load metrics from file
        current_metrics = tester.load_metrics(args.metrics_file)
    else:
        # Run evaluation to generate metrics
        logger.info("No metrics file provided. Running evaluation...")
        try:
            from evaluate_nlu import evaluate_model
            current_metrics = evaluate_model(args.benchmark, args.model, args.output_dir)
        except ImportError:
            logger.error("Could not import evaluate_nlu module. Please provide a metrics file.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            sys.exit(1)

    if not current_metrics:
        logger.error("Failed to load or generate metrics. Exiting.")
        sys.exit(1)

    # Check for regression
    has_regressed, regression_result = tester.check_for_regression(current_metrics)

    # Generate and print report
    report = tester.generate_report(regression_result, args.format)
    print(report)

    # In CI mode, exit with error code if regression detected
    if args.ci and has_regressed:
        logger.error("Performance regression detected. Failing CI.")
        sys.exit(1)

    # Success
    sys.exit(0)

if __name__ == "__main__":
    main() 