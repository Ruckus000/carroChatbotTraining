# /Users/jphilistin/Documents/Coding/carroChatbotTraining/chatbot/test_phase5.py
import os
import sys
import glob
from typing import List, Dict, Set, Tuple, Optional

# List of files that must be present for a successful deployment
required_files = [
    "train.py",
    "inference.py",
    "api.py",
    "dialog_manager.py",
    "response_generator.py",
    "requirements.txt",
    # README can be in root or docs directory
    {"paths": ["README.md", "docs/README.md"], "min_required": 1, "name": "README documentation"},
    ".gitignore",
    "test_integration.py",
    "test_dialog_manager_unified.py",
    "test_api_integration.py",
    # "test_phase5.py" # The test itself
]

# Required directories
required_directories = [
    "data",
    "trained_nlu_model",
    "docs"
]

# Data files that should exist (at least one of these patterns should match)
required_data_patterns = [
    "data/*training_data*.json",   # Any training data JSON file
    "data/nlu_training_data.json", # Specific training data file
]

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {filepath} exists")
        return True
    else:
        print(f"✗ {filepath} is missing")
        return False

def check_alternatives_exist(options: Dict) -> bool:
    """
    Check if at least min_required of the alternative file paths exist.
    Prints appropriate status messages.
    """
    paths = options["paths"]
    min_required = options.get("min_required", 1)
    name = options.get("name", ", ".join(paths))

    existing = [path for path in paths if os.path.exists(path)]

    if len(existing) >= min_required:
        print(f"✓ {name} exists ({', '.join(existing)})")
        return True
    else:
        print(f"✗ {name} is missing (need {min_required} of {paths})")
        return False

def check_pattern_exists(pattern: str) -> bool:
    """Check if any files match the given glob pattern."""
    matches = glob.glob(pattern)
    if matches:
        print(f"✓ Pattern '{pattern}' matches: {', '.join(matches)}")
        return True
    else:
        print(f"✗ Pattern '{pattern}' has no matches")
        return False

def main() -> bool:
    """
    Check that all required deployment files exist.
    Returns True if all files exist, False otherwise.
    """
    print("\n========== Testing Deployment Structure ==========\n")

    all_exist = True
    missing_components: List[str] = []

    # Check each required file
    for item in required_files:
        if isinstance(item, dict):
            # Handle alternative files where we need at least one to exist
            if not check_alternatives_exist(item):
                all_exist = False
                missing_components.append(item.get("name", str(item["paths"])))
        else:
            # Simple file path
            if not check_file_exists(item):
                all_exist = False
                missing_components.append(item)

    # Check for required directories
    for directory in required_directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ Directory {directory}/ exists")
        else:
            print(f"✗ Directory {directory}/ is missing")
            all_exist = False
            missing_components.append(f"{directory}/")

    # Check for required data file patterns
    data_pattern_exists = False
    for pattern in required_data_patterns:
        if check_pattern_exists(pattern):
            data_pattern_exists = True
            break

    if not data_pattern_exists:
        all_exist = False
        missing_components.append("Training data files")
        print(f"✗ No training data files found matching any pattern: {required_data_patterns}")

    # Final result
    print("\n========== Deployment Structure Test Results ==========")
    if all_exist:
        print("✓ SUCCESS: All required files and directories present")
        return True
    else:
        print(f"✗ FAILURE: {len(missing_components)} required components are missing:")
        for i, component in enumerate(missing_components):
            print(f"  {i+1}. {component}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)  # Exit with error code if any files are missing 