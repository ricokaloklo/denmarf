#!/usr/bin/env python

from distutils.core import setup

setup(name='denmarf',
      version='0.3.1',
      description='Density EstimatioN using Masked AutoRegressive Flow',
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
