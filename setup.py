#!/usr/bin/env python

from distutils.core import setup

setup(name='tableshift',
      version='0.0',
      description='A tabular data benchmarking toolkit.',
      author='Josh Gardner',
      author_email='jpgard@cs.washington.edu',
      packages=['tableshift'],
      data_files=[('tableshift/datasets',
                   ['tableshift/datasets/nhanes_data_sources.json',
                    'tableshift/datasets/icd9-codes.json'])]
      )
