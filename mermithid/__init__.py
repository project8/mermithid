'''
'''

from __future__ import absolute_import

import pkg_resources
__version__ = pkg_resources.require("mermithid")[0].version.split('-')[0]
__commit__ = pkg_resources.require("mermithid")[0].version.split('-')[-1]

# from . import misc, processors

__all__ = []
import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)
    for name, value in inspect.getmembers(module):
        if name.startswith("__"):
          continue
        
        globals()[name] = value
        __all__.append(name)
