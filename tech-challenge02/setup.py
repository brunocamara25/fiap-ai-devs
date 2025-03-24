# setup.py
from setuptools import setup, find_packages

setup(
    name="portfolio_optimization",
    version="0.1.0",
    description="Otimização de Portfólio de Investimento usando Algoritmo Genético",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.2.2",
        "deap>=1.3.3",
        "yfinance>=0.2.28",
        "tqdm>=4.66.1",
        "scipy>=1.10.1",
        "plotly>=5.14.1",
        "streamlit>=1.27.0",
        "pytest>=7.3.1"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "jupyter>=1.0.0",
            "black",
            "flake8",
        ],
    },
)