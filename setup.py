#!/usr/bin/env python

from distutils.core import setup
from pathlib import Path

setup(name='denmarf',
      version='0.3.2',
      description='Density EstimatioN using Masked AutoRegressive Flow',
      long_description=Path("README.md").read_text(encoding="utf-8"),
      long_description_content_type="text/markdown",
      author='Rico Ka Lok Lo',
      author_email='rico.kaloklo@gmail.com',
      packages=['denmarf'],
      install_requires=[
            'numpy',
            'scipy',
            'tqdm',
            'torch',
            'getdist',
      ],
)
