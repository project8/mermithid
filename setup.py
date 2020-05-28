#! /usr/bin/env python
from setuptools import find_packages, setup
from glob import glob

import sys
import os
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

# Change tag and version number below
MORPHO_TAG = "v2.5.0"
MORPHO_HASH = 'fb65789' #'0f57cdc'

# Don't touch below
MORPHO_VERSION='{}-15-g{}'.format(MORPHO_TAG,MORPHO_HASH[:7])
MORPHO_DEP_LINK = 'git+https://github.com/morphoorg/morpho.git@{0}#egg=morpho-{1}'.format(MORPHO_HASH,MORPHO_VERSION)
MORPHO_REQ = "morpho=={0}".format(MORPHO_VERSION)

requirements = []
extras_require = {
    'core':['colorlog', MORPHO_REQ],
    'doc': ['sphinx','sphinx_rtd_theme','sphinxcontrib-programoutput', 'six', 'colorlog', MORPHO_REQ]
}
dep_links = {
    MORPHO_DEP_LINK
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
    dependency_links=dep_links,
    url='http://www.github.com/project8/mermithid',
    author = "M. Guigue",
    maintainer = "M. Guigue (PNNL)",
    maintainer_email = "mathieu.guigue@pnnl.gov"
)
