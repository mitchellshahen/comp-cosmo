"""
Set up file for acquiring and installing all the necessary packages to run the scripts and
functions in this environment.

:title: setup.py

:author: Mitchell Shahen

:history: 26/05/2021
"""

from setuptools import setup, find_packages

setup(
    name='chemistrygym',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pickle',
        'scipy',
    ],
    description='Computational Techniques in Cosmology and Stellar Physics',
    author='Mitchell Shahen',
    url='https://github.com/mitchellshahen/comp-cosmo',
    version='0.0'
)
