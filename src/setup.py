# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "Cython==0.29.24"
        "dash==2.0.0"
        "dash-bootstrap-components==1.0.2"
        "dash-core-components==2.0.0"
        "dash-daq==0.3.1"
        "dash-devices==0.1.3"
        "dash-html-components==2.0.0"
        "dash-renderer==1.9.1"
        "dash-table==5.0.0"
        "fire"
        "graphviz==0.18"
        "Grid2Op==1.6.3"
        "gym==0.18.3"
        "gunicorn"
        "matplotlib==3.4.2"
        "matplotlib-inline==0.1.2"
        "numba==0.53.1"
        "pandapower==2.4.0"
        "pandas==1.1.4"
        "Pillow==8.2.0"
        "plotly==5.4.0"
        "psutil"
        "seaborn"
        "scikit-learn"
        "sklearn==0.0"
        "sympy==1.8"
        "tabulate==0.8.9"
        "-f https://download.pytorch.org/whl/torch_stable.html"
        "torch==1.9.0+cpu"
        "tensorboard"
        "tqdm==4.62.0"
        "virtualenv==20.3.1"
        "whitenoise==5.0.1"
    ]
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='interactive-rbg2sr',
      version='0.1.0',
      description='An interactive platform for symbolic regression using plotly and dash',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 0 - Pre-Alpha',
          'Programming Language :: Python :: 3.8',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords=['reinforcement-learning', 'visualization', 'interactive-learning', "preference-learning"
                'rbg2-sr', 'grid2op',
                'powergrid'],
      author='Laure CROCHEPIERRE',
      author_email='laure.crochepierre@rte-france.com',
      url="https://github.com/laure-crochepierre/interactive-rbg2sr",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'interactive_rbg2sr=src.app:run_server'
          ]
      }
      )