from setuptools import find_packages, setup

setup(
    name="MLC",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn", "tqdm"],
)
