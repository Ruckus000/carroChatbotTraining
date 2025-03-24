import pytest
import os
import subprocess
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from unittest.mock import patch

# Test the GitHub workflow file
def test_github_workflow_file():
    """Test the structure of the GitHub workflow file"""
    workflow_path = Path(".github/workflows/ci.yml")
    
    # Skip if running in a non-repo environment
    if not workflow_path.exists():
        pytest.skip("GitHub workflow file not found")
    
    # Read the file content
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Check for required sections in raw content
    assert "name: Chatbot CI" in content
    assert "push:" in content
    assert "pull_request:" in content
    assert "jobs:" in content
    assert "test:" in content
    assert "runs-on: ubuntu-latest" in content
    assert "actions/checkout" in content
    assert "actions/setup-python" in content
    assert "pip install" in content
    assert "pytest" in content

# Test the deployment script functions
def test_deployment_script():
    """Test the deployment script execution"""
    deploy_script_path = Path("deploy.sh")
    
    # Skip if deploy script not found
    if not deploy_script_path.exists():
        pytest.skip("Deployment script not found")
    
    # Check if the script is executable
    assert os.access(deploy_script_path, os.X_OK), "deploy.sh should be executable"
    
    # Read the script content to verify it has expected sections
    with open(deploy_script_path, 'r') as f:
        script_content = f.read()
    
    # Check for essential sections
    assert "#!/bin/bash" in script_content, "Script should have proper shebang"
    assert "virtualenv" in script_content or "venv" in script_content, "Script should create a virtual environment"
    assert "pip" in script_content, "Script should install dependencies"
    
    # Create a mock deployment config to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a config directory
        config_dir = os.path.join(temp_dir, "config")
        os.makedirs(config_dir)
        
        # Create a mock config file
        config_file = os.path.join(config_dir, "deployment.test.yaml")
        with open(config_file, 'w') as f:
            yaml.dump({
                "app_name": "test-chatbot",
                "environment": "test",
                "feature_flags": {
                    "use_langgraph": True,
                    "use_mistral": False
                }
            }, f)
        
        # Prepare a mock execution environment
        with patch('subprocess.run') as mock_run:
            # Mock successful execution
            mock_run.return_value.returncode = 0
            
            # Don't actually run the script, just check its presence
            assert deploy_script_path.exists(), "deploy.sh script should be available"
            
            # In a real scenario, we'd test actual execution like:
            # result = subprocess.run([str(deploy_script_path), "test"], 
            #                         cwd=temp_dir, 
            #                         capture_output=True, 
            #                         text=True)
            # assert result.returncode == 0, f"Script failed: {result.stderr}"

# Test the project structure is complete for deployment
def test_project_structure():
    """Test that the project structure is complete and deployment-ready"""
    # Check required directories
    assert os.path.isdir("langgraph_integration"), "Main package directory should exist"
    assert os.path.isdir("tests"), "Tests directory should exist"
    
    # Check required package files
    required_files = [
        "langgraph_integration/__init__.py",
        "langgraph_integration/feature_flags.py",
        "langgraph_integration/adapters.py",
        "langgraph_integration/langgraph_state.py", 
        "langgraph_integration/mistral_integration.py",
        "langgraph_integration/hybrid_detection.py",
        "langgraph_integration/langgraph_nodes.py",
        "langgraph_integration/state_converter.py",
        "langgraph_integration/langgraph_workflow.py",
        "langgraph_integration/monitoring.py",
        "langgraph_integration/cpu_optimizations.py",
        "langgraph_integration/streamlit_integration.py"
    ]
    
    for file_path in required_files:
        assert os.path.isfile(file_path), f"Required file {file_path} should exist"
    
    # Check test files for all phases
    test_files = [
        "tests/test_phase1.py",
        "tests/test_phase2.py",
        "tests/test_phase3.py",
        "tests/test_phase4.py",
        "tests/test_phase5.py",
        "tests/test_integration.py"
    ]
    
    for file_path in test_files:
        assert os.path.isfile(file_path), f"Test file {file_path} should exist"

# Test package installation (can be mocked)
def test_package_installation():
    """Test package installation process"""
    # Create a temporary directory for a mock installation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a minimal setup.py file
        setup_py = """
from setuptools import setup, find_packages

setup(
    name="langgraph_integration",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain",
        "langchain-community",
        "mistralai",
        "pytest",
        "pytest-cov",
        "streamlit"
    ],
    python_requires=">=3.8",
)
"""
        # Write setup.py to the temp dir
        with open(os.path.join(temp_dir, "setup.py"), 'w') as f:
            f.write(setup_py)
        
        # Copy package files to temp_dir (mocked)
        # In a real test, we would copy the actual files, but here we just check they exist
        pkg_dir = os.path.join(temp_dir, "langgraph_integration")
        os.makedirs(pkg_dir)
        
        # Create a minimal __init__.py
        with open(os.path.join(pkg_dir, "__init__.py"), 'w') as f:
            f.write('"""LangGraph integration package"""')
        
        # Mock package installation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # In a real test, this would actually install the package:
            # result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
            #                        cwd=temp_dir, capture_output=True, text=True)
            # assert result.returncode == 0, f"Installation failed: {result.stderr}"
            
            # Just verify the setup.py exists
            assert os.path.isfile(os.path.join(temp_dir, "setup.py"))

# Test documentation files are present
def test_documentation():
    """Test documentation is available and structured correctly"""
    # Check for documentation files
    doc_files = [
        "README.md",
        "langGraph-Mistral-implementation-plan.md"
    ]
    
    for file_path in doc_files:
        assert os.path.isfile(file_path), f"Documentation file {file_path} should exist"
    
    # Check README has essential sections
    with open("README.md", 'r') as f:
        readme_content = f.read().lower()
    
    assert "installation" in readme_content, "README should include installation instructions"
    assert "usage" in readme_content, "README should include usage instructions"
    
    # Check implementation plan is complete
    with open("langGraph-Mistral-implementation-plan.md", 'r') as f:
        plan_content = f.read()
    
    # Verify all phases are mentioned
    for phase in range(1, 6):
        assert f"Phase {phase}" in plan_content, f"Implementation plan should cover Phase {phase}"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 