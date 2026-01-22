#! /usr/bin/env python
from setuptools import find_packages, setup
from glob import glob

import sys
import os
from setuptools.command.test import test as TestCommand

verstr = "none"
try:
    import subprocess
    #Modified this to comply with pip's current PEP requirements (versioning conventions)
    version_info = subprocess.check_output(['git','describe', '--long']).decode('utf-8').strip() 
    version_info = version_info.split('-')
    verstr = version_info[0] + '+' + version_info[1] + '.' + version_info[2]
except EnvironmentError:
    pass
except Exception as err:
    print(err)
    verstr = 'v0.0.0+???'

on_rtd = os.environ.get("READTHEDOCS", None) == 'True'

requirements = []
extras_require = {
    'core':['colorlog', 'iminuit'],
    'doc': ['sphinx','sphinx_rtd_theme','sphinxcontrib-programoutput', 'six', 'colorlog']
}

if on_rtd:
    requirements.append('better-apidoc')
    requirements += extras_require['doc']
else:
    requirements = extras_require['core']

everything = set()
for deps in extras_require.values():
    everything.update(deps)
extras_require['all'] = everything

setup(
    name='mermithid',
    version=verstr,
    description="A Project 8 extension to morpho",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    url='http://www.github.com/project8/mermithid',
    author = "M. Guigue",
    maintainer = "T. E. Weiss (Yale)",
    maintainer_email = "talia.weiss@yale.edu"
)
