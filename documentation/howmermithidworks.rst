-------------------------------------
How does mermithid work?
-------------------------------------

New processors
------------------------

Mermithid extends morpho by adding new processors accessible to morpho's ToolBox class.
The new processors are created similarly to the one in morpho (aka they inheritates from the ``BaseProcessor`` class and should implement ``InternalConfigure`` and ``InternalRun`` method): a tutorial explaining how processors are created in morpho is available `here`_.

.. _here: https://morpho.readthedocs.io/en/latest/morpho2newprocessors.html

New classes and functions are added to the mermithid package without actions from the developer and are accessible via import or in configuration file (for example, the ``ProcessorX`` is accesible as ``mermithid:ProcessorX``).


Interface with C++ code
------------------------

In order to keep morpho free from experiment-specific tools, external libraries should added in a derived package (such as mermithid).
In the case of mermithid, there are currently only C++ libraries (naming `Cicada`_ and `Phylloxera`_).
These libraries contain new TObject-derived classes: thanks to that, loading them via ``gSystem`` using dedicated python scripts (`example`_) that allows us to access to the new classes in pyROOT (via a ``from ROOT import ProcessorX``).

.. _Cicada: https://p8-cicada.readthedocs.io/en/latest
.. _Phylloxera: https://github.com/project8/phylloxera
.. _example: https://github.com/project8/phylloxera/blob/master/src/PhylloxeraPy.py
