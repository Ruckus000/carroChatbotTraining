#!/usr/bin/env python3
"""
Intelligent NLU Model Pipeline - Training and benchmarking pipeline
with model versioning, metadata tracking, and quality gates.
"""

import os
import sys
import json
import uuid
import shutil
import argparse
import logging
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nlu_pipeline')

class ModelPipeline:
    """
    Intelligent NLU model pipeline with version tracking and quality gates.
    """

    def __init__(self, base_dir: str = "."):
        """
        Initialize the pipeline with directory structure.

        Args:
            base_dir: Base directory for the pipeline
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.benchmark_dir = self.base_dir / "benchmark_results"
        self.data_dir = self.base_dir / "data"

        # Create required directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.benchmark_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize model registry file
        self.registry_path = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load or initialize the model registry"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading model registry: {str(e)}. Creating new one.")

        # Initialize new registry
        registry = {
            "models": {},
            "current_model": None,
            "best_model": None,
            "last_updated": datetime.now().isoformat()
        }

        # Save initial registry
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return registry

    def _save_registry(self):
        """Save the model registry to disk"""
        self.registry["last_updated"] = datetime.now().isoformat()

        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _generate_model_id(self, training_data_path: str) -> str:
        """
        Generate a unique model ID based on timestamp and data hash.

        Args:
            training_data_path: Path to training data

        Returns:
            str: Unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a hash of the training data for deterministic versioning
        data_hash = "unknown"
        if os.path.exists(training_data_path):
            try:
                with open(training_data_path, 'rb') as f:
                    data_hash = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                pass

        return f"model_{timestamp}_{data_hash}"

    def _calculate_data_stats(self, data_path: str) -> Dict[str, Any]:
        """
        Calculate statistics about the training data.

        Args:
            data_path: Path to training data

        Returns:
            Dict[str, Any]: Data statistics
        """
        stats = {
            "file_size_bytes": 0,
            "example_count": 0,
            "intent_distribution": {},
            "entity_distribution": {},
            "avg_tokens_per_example": 0
        }

        if not os.path.exists(data_path):
            return stats

        try:
            with open(data_path, 'r') as f:
                data = json.load(f)

            stats["file_size_bytes"] = os.path.getsize(data_path)
            stats["example_count"] = len(data)

            total_tokens = 0

            # Calculate distributions
            for example in data:
                # Intent distribution
                intent = example.get('intent')
                if intent:
                    stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1

                # Entity distribution
                for entity in example.get('entities', []):
                    entity_type = entity.get('entity')
                    if entity_type:
                        stats["entity_distribution"][entity_type] = stats["entity_distribution"].get(entity_type, 0) + 1

                # Token count
                tokens = example.get('text', '').split()
                total_tokens += len(tokens)

            # Calculate average tokens
            if stats["example_count"] > 0:
                stats["avg_tokens_per_example"] = total_tokens / stats["example_count"]

        except Exception as e:
            logger.warning(f"Error calculating data stats: {str(e)}")

        return stats

    def run_command(self, command: str) -> Tuple[int, str]:
        """
        Run a shell command with output capture and logging.

        Args:
            command: Command to run

        Returns:
            Tuple[int, str]: (return_code, output)
        """
        logger.info(f"Running: {command}")

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Capture output with streaming
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            logger.debug(line)
            output_lines.append(line)

        # Wait for process to complete
        return_code = process.wait()
        output = '\n'.join(output_lines)

        return return_code, output

    def train(
        self,
        training_data_path: str,
        model_id: Optional[str] = None,
        description: str = "",
        train_args: Optional[List[str]] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Train a new model and track its metadata.

        Args:
            training_data_path: Path to training data
            model_id: Optional custom model ID
            description: Optional model description
            train_args: Additional arguments for train.py
            force: Force training even if data hasn't changed

        Returns:
            Tuple[bool, str]: (success, model_id)
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(training_data_path)

        # Ensure model_id is safe for filesystem
        model_id = model_id.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Define model directory
        model_dir = self.models_dir / model_id

        # Check if model with this ID already exists
        if os.path.exists(model_dir) and not force:
            logger.warning(f"Model with ID {model_id} already exists. Use force=True to overwrite.")
            return False, model_id

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Calculate data statistics
        data_stats = self._calculate_data_stats(training_data_path)

        # Create metadata file
        metadata = {
            "model_id": model_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "training_data": {
                "path": str(training_data_path),
                "stats": data_stats
            },
            "train_args": train_args or [],
            "metrics": None,
            "status": "training"
        }

        # Save initial metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Build training command
        train_cmd = [
            "python", "train.py",
            "--data", str(training_data_path),
            "--output", str(model_dir)
        ]

        # Add additional arguments
        if train_args:
            train_cmd.extend(train_args)

        # Run training
        train_command = " ".join(train_cmd)
        return_code, output = self.run_command(train_command)

        # Update metadata with training results
        metadata["train_output"] = output

        if return_code != 0:
            metadata["status"] = "failed"
            logger.error(f"Training failed with code {return_code}")

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return False, model_id

        # Training succeeded
        metadata["status"] = "trained"

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["models"][model_id] = {
            "path": str(model_dir),
            "created_at": metadata["created_at"],
            "description": description,
            "status": "trained"
        }

        # Set as current model
        self.registry["current_model"] = model_id
        self._save_registry()

        logger.info(f"Successfully trained model: {model_id}")
        return True, model_id

    def benchmark(
        self,
        model_id: Optional[str] = None,
        benchmark_data_path: Optional[str] = None,
        benchmark_args: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Benchmark a model and store its metrics.

        Args:
            model_id: Model ID to benchmark (uses current if None)
            benchmark_data_path: Path to benchmark data
            benchmark_args: Additional arguments for evaluate_nlu.py

        Returns:
            Tuple[bool, Dict]: (success, metrics)
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False, {}

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False, {}

        # Get model directory
        model_dir = Path(self.registry["models"][model_id]["path"])

        # Default benchmark data path
        if benchmark_data_path is None:
            benchmark_data_path = str(self.data_dir / "nlu_benchmark_data.json")

        # Create benchmark output directory
        benchmark_output_dir = self.benchmark_dir / model_id
        os.makedirs(benchmark_output_dir, exist_ok=True)

        # Build benchmark command
        benchmark_cmd = [
            "python", "evaluate_nlu.py",
            "--benchmark", str(benchmark_data_path),
            "--model", str(model_dir),
            "--output", str(benchmark_output_dir)
        ]

        # Add additional arguments
        if benchmark_args:
            benchmark_cmd.extend(benchmark_args)

        # Run benchmark
        benchmark_command = " ".join(benchmark_cmd)
        return_code, output = self.run_command(benchmark_command)

        if return_code != 0:
            logger.error(f"Benchmarking failed with code {return_code}")
            return False, {}

        # Find the metrics file
        metrics_files = list(benchmark_output_dir.glob("metrics_*.json"))
        if not metrics_files:
            logger.error("No metrics file found after benchmarking.")
            return False, {}

        # Get the latest metrics file
        latest_metrics_file = max(metrics_files, key=os.path.getctime)

        # Load metrics
        try:
            with open(latest_metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics file: {str(e)}")
            return False, {}

        # Update model metadata with metrics
        metadata_path = model_dir / "metadata.json"

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata["metrics"] = metrics
            metadata["benchmarked_at"] = datetime.now().isoformat()
            metadata["benchmark_data_path"] = str(benchmark_data_path)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update model metadata with metrics: {str(e)}")

        # Update registry
        self.registry["models"][model_id]["benchmarked_at"] = datetime.now().isoformat()
        self.registry["models"][model_id]["metrics_path"] = str(latest_metrics_file)

        # Extract key metrics for registry
        key_metrics = {}
        intent_metrics = metrics.get("intent_metrics", {})
        entity_metrics = metrics.get("entity_metrics", {})

        key_metrics["intent_f1"] = intent_metrics.get("f1", 0.0)
        key_metrics["accuracy"] = intent_metrics.get("accuracy", 0.0)

        if isinstance(entity_metrics, dict) and "micro avg" in entity_metrics:
            key_metrics["entity_f1"] = entity_metrics["micro avg"].get("f1-score", 0.0)

        self.registry["models"][model_id]["key_metrics"] = key_metrics

        # Check if this is the best model
        if self._is_best_model(model_id):
            self.registry["best_model"] = model_id
            logger.info(f"New best model: {model_id}")

        self._save_registry()

        logger.info(f"Successfully benchmarked model: {model_id}")
        return True, metrics

    def _is_best_model(self, model_id: str) -> bool:
        """
        Check if the given model is the best model based on metrics.

        Args:
            model_id: Model ID to check

        Returns:
            bool: True if this is the best model
        """
        # Get current best model
        best_model_id = self.registry.get("best_model")

        # If no best model yet, this is the best
        if best_model_id is None:
            return True

        # Get metrics for both models
        current_metrics = self.registry["models"][model_id].get("key_metrics", {})
        best_metrics = self.registry["models"].get(best_model_id, {}).get("key_metrics", {})

        # Compare intent_f1 (primary metric)
        current_f1 = current_metrics.get("intent_f1", 0.0)
        best_f1 = best_metrics.get("intent_f1", 0.0)

        return current_f1 > best_f1

    def check_regression(
        self,
        model_id: Optional[str] = None,
        config_path: Optional[str] = None,
        ci_mode: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a model has regressed compared to the best model.

        Args:
            model_id: Model ID to check (uses current if None)
            config_path: Path to regression test configuration
            ci_mode: Whether to run in CI mode

        Returns:
            Tuple[bool, Dict]: (has_regressed, details)
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False, {}

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False, {}

        # Get metrics path
        metrics_path = self.registry["models"][model_id].get("metrics_path")
        if not metrics_path:
            logger.error(f"No metrics found for model {model_id}. Run benchmark first.")
            return False, {}

        # Build regression test command
        regression_cmd = [
            "python", "test_nlu_regression.py",
            "--metrics-file", str(metrics_path),
            "--history-file", str(self.benchmark_dir / "metrics_history.csv")
        ]

        # Add config path if provided
        if config_path:
            regression_cmd.extend(["--config", str(config_path)])

        # Add CI mode if enabled
        if ci_mode:
            regression_cmd.append("--ci")

        # Add format
        regression_cmd.extend(["--format", "json"])

        # Run regression test
        regression_command = " ".join(regression_cmd)
        return_code, output = self.run_command(regression_command)

        # Parse JSON output
        try:
            # Find JSON in output (may have other text mixed in)
            json_start = output.find('{')
            json_end = output.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                regression_result = json.loads(json_str)
            else:
                # Fallback for when output is not proper JSON
                regression_result = {"has_regressed": return_code != 0}
        except Exception as e:
            logger.warning(f"Failed to parse regression test output: {str(e)}")
            regression_result = {"has_regressed": return_code != 0}

        # Update model metadata with regression result
        try:
            model_dir = Path(self.registry["models"][model_id]["path"])
            metadata_path = model_dir / "metadata.json"

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata["regression_test"] = {
                "result": regression_result,
                "ran_at": datetime.now().isoformat()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update model metadata with regression result: {str(e)}")

        has_regressed = regression_result.get("has_regressed", False)

        if has_regressed:
            logger.warning(f"Model {model_id} has regressed!")
        else:
            logger.info(f"Model {model_id} has not regressed.")

        return has_regressed, regression_result

    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models in the registry.

        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        model_list = []

        for model_id, model_data in self.registry["models"].items():
            # Determine if this is the current or best model
            is_current = model_id == self.registry.get("current_model")
            is_best = model_id == self.registry.get("best_model")

            # Add model to list
            model_list.append({
                "model_id": model_id,
                "path": model_data["path"],
                "created_at": model_data.get("created_at"),
                "description": model_data.get("description", ""),
                "status": model_data.get("status", "unknown"),
                "key_metrics": model_data.get("key_metrics", {}),
                "is_current": is_current,
                "is_best": is_best
            })

        # Sort by creation date (newest first)
        model_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return model_list

    def set_current_model(self, model_id: str) -> bool:
        """
        Set the current model for inference.

        Args:
            model_id: Model ID to set as current

        Returns:
            bool: Success
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False

        # Update registry
        self.registry["current_model"] = model_id
        self._save_registry()

        logger.info(f"Set current model to: {model_id}")
        return True

    def export_model(self, model_id: Optional[str] = None, export_dir: str = "export") -> bool:
        """
        Export a model for deployment.

        Args:
            model_id: Model ID to export (uses current if None)
            export_dir: Directory to export to

        Returns:
            bool: Success
        """
        # Use current model if not specified
        if model_id is None:
            model_id = self.registry.get("current_model")
            if model_id is None:
                logger.error("No current model found. Train a model first.")
                return False

        # Check if model exists
        if model_id not in self.registry["models"]:
            logger.error(f"Model with ID {model_id} not found in registry.")
            return False

        # Get model directory
        model_dir = Path(self.registry["models"][model_id]["path"])

        # Create export directory
        export_path = Path(export_dir)
        os.makedirs(export_path, exist_ok=True)

        try:
            # Copy model files
            shutil.copytree(
                model_dir,
                export_path / model_id,
                dirs_exist_ok=True
            )

            # Create symlink to 'latest'
            latest_link = export_path / "latest"
            if os.path.exists(latest_link):
                os.remove(latest_link)

            # Create relative symlink
            os.symlink(model_id, latest_link, target_is_directory=True)

            logger.info(f"Exported model {model_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            return False

    def run_full_pipeline(
        self,
        training_data_path: str,
        benchmark_data_path: str,
        description: str = "",
        train_args: Optional[List[str]] = None,
        benchmark_args: Optional[List[str]] = None,
        regression_config_path: Optional[str] = None,
        export_dir: Optional[str] = None,
        fail_on_regression: bool = False
    ) -> Tuple[bool, str]:
        """
        Run the full model pipeline: train, benchmark, regression test, export.

        Args:
            training_data_path: Path to training data
            benchmark_data_path: Path to benchmark data
            description: Model description
            train_args: Additional arguments for train.py
            benchmark_args: Additional arguments for evaluate_nlu.py
            regression_config_path: Path to regression test configuration
            export_dir: Directory to export to if successful
            fail_on_regression: Whether to fail if regression detected

        Returns:
            Tuple[bool, str]: (success, model_id)
        """
        logger.info("Starting full model pipeline")

        # Step 1: Train model
        logger.info("Step 1: Training model")
        train_success, model_id = self.train(
            training_data_path=training_data_path,
            description=description,
            train_args=train_args
        )

        if not train_success:
            logger.error("Pipeline failed at training step")
            return False, model_id

        # Step 2: Benchmark model
        logger.info("Step 2: Benchmarking model")
        benchmark_success, metrics = self.benchmark(
            model_id=model_id,
            benchmark_data_path=benchmark_data_path,
            benchmark_args=benchmark_args
        )

        if not benchmark_success:
            logger.error("Pipeline failed at benchmarking step")
            return False, model_id

        # Step 3: Regression test
        logger.info("Step 3: Running regression test")
        has_regressed, regression_details = self.check_regression(
            model_id=model_id,
            config_path=regression_config_path
        )

        if has_regressed and fail_on_regression:
            logger.error("Pipeline failed due to regression")
            return False, model_id

        # Step 4: Export model if requested
        if export_dir:
            logger.info("Step 4: Exporting model")
            export_success = self.export_model(
                model_id=model_id,
                export_dir=export_dir
            )

            if not export_success:
                logger.error("Pipeline failed at export step")
                return False, model_id

        logger.info(f"Full pipeline completed successfully. Model ID: {model_id}")
        return True, model_id

def main():
    """Main entry point for the model pipeline"""
    parser = argparse.ArgumentParser(description='NLU Model Pipeline')
    parser.add_argument('--base-dir', default=".", help='Base directory for pipeline')

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--model-id', help='Custom model ID')
    train_parser.add_argument('--description', default="", help='Model description')
    train_parser.add_argument('--force', action='store_true', help='Force training')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark a model')
    benchmark_parser.add_argument('--model-id', help='Model ID to benchmark (uses current if not specified)')
    benchmark_parser.add_argument('--data', help='Path to benchmark data')

    # Regression command
    regression_parser = subparsers.add_parser('regression', help='Run regression test')
    regression_parser.add_argument('--model-id', help='Model ID to test (uses current if not specified)')
    regression_parser.add_argument('--config', help='Path to regression test configuration')
    regression_parser.add_argument('--ci', action='store_true', help='Run in CI mode')

    # List command
    list_parser = subparsers.add_parser('list', help='List all models')
    list_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export a model')
    export_parser.add_argument('--model-id', help='Model ID to export (uses current if not specified)')
    export_parser.add_argument('--export-dir', default='export', help='Directory to export to')

    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--training-data', required=True, help='Path to training data')
    pipeline_parser.add_argument('--benchmark-data', required=True, help='Path to benchmark data')
    pipeline_parser.add_argument('--description', default="", help='Model description')
    pipeline_parser.add_argument('--regression-config', help='Path to regression test configuration')
    pipeline_parser.add_argument('--export-dir', help='Directory to export to if successful')
    pipeline_parser.add_argument('--fail-on-regression', action='store_true', help='Fail if regression detected')

    # Parse arguments
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ModelPipeline(args.base_dir)

    # Run command
    if args.command == 'train':
        success, model_id = pipeline.train(
            training_data_path=args.data,
            model_id=args.model_id,
            description=args.description,
            force=args.force
        )

        if success:
            print(f"Successfully trained model: {model_id}")
        else:
            print(f"Failed to train model: {model_id}")
            sys.exit(1)

    elif args.command == 'benchmark':
        success, metrics = pipeline.benchmark(
            model_id=args.model_id,
            benchmark_data_path=args.data
        )

        if success:
            # Print key metrics
            intent_metrics = metrics.get("intent_metrics", {})
            entity_metrics = metrics.get("entity_metrics", {})

            print("\nKey Metrics:")
            print(f"Intent Accuracy: {intent_metrics.get('accuracy', 0):.4f}")
            print(f"Intent F1 Score: {intent_metrics.get('f1', 0):.4f}")

            if isinstance(entity_metrics, dict) and "micro avg" in entity_metrics:
                print(f"Entity F1 Score: {entity_metrics['micro avg'].get('f1-score', 0):.4f}")
        else:
            print("Failed to benchmark model")
            sys.exit(1)

    elif args.command == 'regression':
        has_regressed, details = pipeline.check_regression(
            model_id=args.model_id,
            config_path=args.config,
            ci_mode=args.ci
        )

        if args.ci and has_regressed:
            sys.exit(1)

    elif args.command == 'list':
        model_list = pipeline.get_model_list()

        if args.json:
            print(json.dumps(model_list, indent=2))
        else:
            print("\nModel Registry:")
            print(f"{'Model ID':<30} {'Status':<10} {'Intent F1':<10} {'Created At':<20} {'Description'}")
            print("-" * 80)

            for model in model_list:
                # Mark current and best models
                marker = ""
                if model["is_current"] and model["is_best"]:
                    marker = "[CURRENT,BEST] "
                elif model["is_current"]:
                    marker = "[CURRENT] "
                elif model["is_best"]:
                    marker = "[BEST] "

                model_id = model["model_id"]
                status = model["status"]
                intent_f1 = model.get("key_metrics", {}).get("intent_f1", 0)
                created_at = model.get("created_at", "")[:19]  # Truncate to date+time
                description = model.get("description", "")

                print(f"{marker}{model_id:<20} {status:<10} {intent_f1:<10.4f} {created_at:<20} {description}")

    elif args.command == 'export':
        success = pipeline.export_model(
            model_id=args.model_id,
            export_dir=args.export_dir
        )

        if not success:
            print("Failed to export model")
            sys.exit(1)

    elif args.command == 'pipeline':
        success, model_id = pipeline.run_full_pipeline(
            training_data_path=args.training_data,
            benchmark_data_path=args.benchmark_data,
            description=args.description,
            regression_config_path=args.regression_config,
            export_dir=args.export_dir,
            fail_on_regression=args.fail_on_regression
        )

        if success:
            print(f"Pipeline completed successfully. Model ID: {model_id}")
        else:
            print(f"Pipeline failed. Model ID: {model_id}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 