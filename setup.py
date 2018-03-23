#! /usr/bin/env python
from setuptools import find_packages, setup
from glob import glob

import sys, os
from setuptools.command.test import test as TestCommand

verstr = "none"
try:
    import subprocess
    verstr = subprocess.check_output(['git','describe', '--long']).decode('utf-8').strip()
except EnvironmentError:
    pass
except Exception as err:
    print(err)
    verstr = 'v0.0.0-???'

on_rtd = os.environ.get("READTHEDOCS", None) == 'True'

requirements = []
extras_require = {
    'core':['morpho','matplotlib==1.5.1','colorlog'],
    'doc': ['sphinx','sphinx_rtd_theme','sphinxcontrib-programoutput']
}

if on_rtd:
    requirements.append('better-apidoc')
    requirements += extras_require['doc']
else:
    requirements = extras_require['core']
    try:
        import CicadaPy
    except ImportError:
        print('\nError! Cicada is required to build from source.')
        print('Please install it and make sure you have added the libraries to your PYTHONPATH.')
        print('Documentation can be found here: ')
        print('  http://p8-cicada.readthedocs.io/en/latest')
        sys.exit(1)
    try:
        import PhylloxeraPy
    except ImportError:
        print('\nError! Phylloxera is required to build from source.')
        print('Please install it and make sure you have added the libraries to your PYTHONPATH.')
        print('Documentation can be found here: ')
        print('  http://github.com/project8/phylloxera')
        sys.exit(1)

everything = set()
for deps in extras_require.values():
    everything.update(deps)
extras_require['all'] = everything

setup(
    name='mermithid',
    version=verstr,
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    url='http://www.github.com/project8/mermithid',
)
