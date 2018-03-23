from setuptools import setup, find_packages
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
    'core':['morpho','pystan','matplotlib==1.5.1','colorlog'],
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
        sys.exit("CicadaPy cannot be imported. Make sure it has been installed and the PYTHONPATH is properly setup.")
    try:
        import PhylloxeraPy
    except ImportError:
        sys.exit("PhylloxeraPy cannot be imported. Make sure it has been installed and the PYTHONPATH is properly setup.")


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
