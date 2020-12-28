#!/usr/bin/env python 
from setuptools import setup

setup(
    name='multipole_dens',
    version='0.0.1',
    author='Aidan C',
    packages=['multipole'],
    url='http://pypi.python.org/pypi/PackageName/',
    description='An awesome package that calculates dipole, quadrapole and charge densities',
    long_description=open('README.md').read(),
    install_requires=[
       "MDAnalysis >= 1.0.0",
       "numpy",
   ],
)
