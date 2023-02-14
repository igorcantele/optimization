from setuptools import setup, find_packages

setup(
    name="optimizer",
    author="Igor Cantele",
    version="0.0.1",
    packages=find_packages(),
    install_requires = ["deap", "numpy", "bmg"]
)