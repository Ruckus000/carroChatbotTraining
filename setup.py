from setuptools import setup, find_packages

setup(
    name="langgraph_integration",
    version="0.1.0",
    description="Integration of LangGraph with Mistral for chatbot capabilities",
    author="Carro Chatbot Team",
    author_email="team@example.com",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.0.11",
        "langchain>=0.0.267",
        "langchain-community>=0.0.6",
        "requests>=2.28.0",
        "streamlit>=1.22.0",
        "pytest>=7.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 