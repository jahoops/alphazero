# /home/j/GIT/nnbattle/setup.py
from setuptools import setup, find_packages

setup(
    name='nnbattle',
    version='0.1',
    packages=find_packages(include=['nnbattle', 'nnbattle.*']),
    install_requires=[
        'pytorch_lightning',
        'torch',
        'numpy'
    ],
)