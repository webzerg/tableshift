#!/usr/bin/env python

from distutils.core import setup

setup(name='tablebench',
      version='0.0',
      description='A tabular data benchmarking toolkit.',
      author='Josh Gardner',
      author_email='jpgard@cs.washington.edu',
      packages=['tablebench'],
      data_files=[('tablebench/datasets',
                   ['tablebench/datasets/nhanes_data_sources.json',
                    'tablebench/datasets/icd9-codes.json'])]
      )
