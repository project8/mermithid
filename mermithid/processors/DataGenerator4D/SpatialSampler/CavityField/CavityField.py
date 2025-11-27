"""
The cavity field class for spatial variable sampling. This class will handle the
setting of the field map and interpolation of the field values. Below is the
related physics background for the trapping efficiency calculation.

    1. Trapping condition is $\\sin(\\theta) > \\sqrt{B_{z}(r, z_{start})/B_{z, max}(r)}$.
    2. It is equivalent to $z_{start} \\in (z_{thr}^{lower} (r, \\theta), z_{thr}^{upper} (r, \\theta))$, where $B_{z}(r, z_{thr}^{lower}) = B_{z}(r, z_{thr}^{upper}) = B_{z, max} (r) \\sin^{2}(\\theta)$.
    3. The electron should be produced uniformly in between $z_{max}^{lower} (r)$ and $z_{max}^{upper} (r)$, where they are the `argmax` of $B_{z}(z_{start} | r)$.
    4. Therefore, the efficiency for an event $(r, \\theta)$ to be trapped is $\\varepsilon (r, \\theta) = \\frac{z_{thr}^{upper}(r, \\theta) - z_{thr}^{lower}(r, \\theta)}{z_{max}^{upper}(r) - z_{max}^{lower}(r)}$.

In conclusion, the CavityField class will provide the following functionalities:
    - Setting the magnetic field map
        - by reading from a file
        - by providing arrays
        - or by calculating analytically
    - Interpolating the field values at given (r, z) positions
    - Calculating the trapping efficiency for given (r, theta) samples

This class is an abstract base class. The child classes should implement the specific
methods for setting the field map.

Author: S. M. Lee
First Date: November 21, 2025
Last Update: November 26, 2025
"""

from __future__ import absolute_import
import abc
import six
from typing import Dict, List, Optional, Union

import numpy as np

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)  # type: ignore


@six.add_metaclass(abc.ABCMeta)
class CavityField:
    """
    The cavity field class for spatial variable sampling. This class will handle the
    setting of the field map and interpolation of the field values.

    Parameters:
        name: The name of the instance
    """

    def __init__(
        self,
        name,
    ):
        # Class information
        self._fieldName = name
        logger.debug("Creating CavityField <{}>".format(self._fieldName))

    @abc.abstractmethod
    def ReadFieldMap(self, path: str) -> bool:
        """
        Read the CCA magnetic field map from a file.

        Results:
            True if successful.
        """
        return False

    # @abc.abstractmethod
    # def GetFieldValue(self, r: float, z: float) -> float:

    @abc.abstractmethod
    def GetEfficiency(
        self,
        r: Union[float, np.ndarray],  # (m)
        theta: Union[float, np.ndarray],  # (rad)
    ) -> np.ndarray:
        """
        Calculate the trapping efficiency for given (r, theta) samples.

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            theta: (N,) shaped array or (N, M) mesh of pitch angle(s) (radians)

        Results:
            (N,) array or (N, M) mesh of the trapping efficiency value(s).
        """
        pass
