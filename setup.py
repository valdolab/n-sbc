from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nsbc",
    version="1.0.0",
    author="Osvaldo Velazquez",
    author_email="osvaldodvego@gmail.com",
    description="n-SBC: A novel machine learning model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valdolab/n-sbc",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0",
            "pre-commit>=3.0.0",
            "ruff>=0.4.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "pandas>=2.0.0",
        ],
    },
)
