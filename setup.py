from setuptools import setup, find_packages

setup(
    name="bandits-lib",
    version="0.2.0",
    author="Gustavo Fonseca",
    author_email="gustavo.fonseca@ga.ita.br",
    description="Package to run multi-armed bandit algorithms related to portfolio optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gfonseca92/bandits-lib",
    packages=find_packages(),
    install_requires=[
        "multi-armed-bandit @ git+https://github.com/gfonseca92/Multi-armed-bandit.git",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",

        # Add other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
