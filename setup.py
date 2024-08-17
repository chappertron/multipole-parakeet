#!/usr/bin/env python
from setuptools import setup

setup(
    name="multipole_dens",
    version="0.2.0",
    author="Aidan C",
    packages=["multipole"],
    scripts=["bin/calc_multipoles_dens"],
    url="http://pypi.python.org/pypi/PackageName/",
    description="An awesome package that calculates dipole, "
    "quadrapole and charge densities",
    long_description=open("README.md").read(),
    install_requires=[
        "MDAnalysis >= 2.0.0",
        "numpy",
        "numba",
        "fast_histogram",
        "scipy",
    ],
)
