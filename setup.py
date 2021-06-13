from setuptools import setup, find_packages
import os

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='Adelie',
    version='0.0.1',
    url='https://github.com/JRBCH/adelie',
    author='Julian Rossbroich',
    author_email='julian.rossbroich@fmi.ch',
    description='Simulating rate-based neuron models for Computational Neuroscience with PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["torch"],
)