from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="shortcut-lens",
    version="0.1.0",
    description="A robustness-auditing framework for detecting spurious-correlation "
    "(shortcut) reliance in classical and modern tabular ML classifiers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sunjoh Abdurazack",
    url="https://github.com/<username>/shortcut-lens",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "scipy>=1.9",
        "statsmodels>=0.13",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "pyyaml>=6.0",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.7"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "flake8", "jupyter"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
