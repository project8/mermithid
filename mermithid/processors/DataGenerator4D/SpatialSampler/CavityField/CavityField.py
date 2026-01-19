"""
The abstract base class for the cavity field, which is used for the trapping
efficiency calculation and the theta_start->center conversion in the spatial
variable sampling. The child classes should implement the specific methods for
setting the field map.

Author: S. M. Lee
First Date: November 21, 2025
Last Update: January 19, 2026
"""

from __future__ import absolute_import
import abc
import six
from typing import Tuple, Union

import numpy as np

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


@six.add_metaclass(abc.ABCMeta)
class CavityField:
    """
    The cavity field class for spatial variable sampling. This class will handle the
    setting of the field map and interpolation of the field values. Below is the
    related physics background for the trapping efficiency calculation.

        1. Trapping condition is $\\sin(\\theta_{start}) > \\sqrt{B(r_{start}, z_{start})/B_{max}(r_{start})}$.
        2. It is equivalent to $z_{start} \\in (z_{thr}^{lower} (r_start, \\theta_start), z_{thr}^{upper} (r_start, \\theta_start))$, where $B(r_{start}, z_{thr}^{lower}) = B(r_{start}, z_{thr}^{upper}) = B_{max} (r_{start}) \\sin^{2}(\\theta_{start})$.
        3. The electron should be produced uniformly in between $z_{max}^{lower} (r)$ and $z_{max}^{upper} (r)$, where they are the `argmax` of $B(z_{start} | r_{start})$.
        4. Therefore, the efficiency for an event $(r_{start}, \\theta_{start})$ to be trapped is $\\varepsilon (r_{start}, \\theta_{start}) = \\frac{z_{thr}^{upper}(r_{start}, \\theta_{start}) - z_{thr}^{lower}(r_{start}, \\theta_{start})}{z_{max}^{upper}(r_{start}) - z_{max}^{lower}(r_{start})}$.

    In conclusion, the CavityField class will provide the following functionalities:
        - Setting the magnetic field map
            - by reading from a file,
            - by providing arrays,
            - or by calculating analytically.
        - Interpolating the field values at given (r, z) positions.
        - Calculating the trapping efficiency for given (r, theta) samples.
        - Calculating the z_argmax for given (r), and z_threshold for given (r, theta).
            - for the efficiency calculation and for the theta_start->center conversion.

    This class is an abstract base class. The child classes should implement the specific
    methods for setting the field map.

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

    @abc.abstractmethod
    def GetFieldValue(
        self,
        r: Union[float, np.ndarray],  # (m)
        z: Union[float, np.ndarray],  # (m)
    ) -> Union[float, np.ndarray]:
        """
        Get the magnetic field value(s) at given (r, z) position(s).

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            z: (N,) shaped array or (N, M) mesh of axial position(s) (m)

        Results:
            (N,) array or (N, M) mesh of the magnetic field value(s) (T).
        """
        raise NotImplementedError("GetFieldValue method not implemented.")

    @abc.abstractmethod
    def GetZArgmax(
        self,
        r: Union[float, np.ndarray],  # (m)
        z_center: float = 0.0,  # (m)
    ) -> np.ndarray:
        """
        Calculate the z position(s) where B(r, z_argmax) = B_max(r).

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            z_center: The center z position (m)

        Results:
            (N, 2) array or (N, M, 2) mesh of the z position(s) (m). The last
            dimension corresponds to (z_max_lower, z_max_upper).
        """
        raise NotImplementedError("GetZArgmax method not implemented.")

    @abc.abstractmethod
    def GetZThreshold(
        self,
        r: Union[float, np.ndarray],  # (m)
        theta: Union[float, np.ndarray],  # (rad)
        z_center: float = 0.0,  # (m)
    ) -> np.ndarray:
        """
        Calculate the z position(s) where B(r, z_thr) = B_max(r) * sin^2(theta).

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            theta: (N,) shaped array or (N, M) mesh of pitch angle(s) (radians)
            z_center: The center z position (m)

        Results:
            (N, 2) array or (N, M, 2) mesh of the z position(s) (m). The last
            dimension corresponds to (z_thr_lower, z_thr_upper).
        """
        raise NotImplementedError("GetZThreshold method not implemented.")

    @abc.abstractmethod
    def GetEfficiency(
        self,
        r: Union[float, np.ndarray],  # (m)
        theta: Union[float, np.ndarray],  # (rad)
        z_center: float = 0.0,  # (m)
        return_z_thr: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the trapping efficiency for given (r, theta) samples.

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            theta: (N,) shaped array or (N, M) mesh of pitch angle(s) (radians)
            z_center: The center z position (m)

        Results:
            (N,) array or (N, M) mesh of the trapping efficiency value(s).
            If return_z_thr is True, also returns the corresponding z_thr values.
        """
        raise NotImplementedError("GetEfficiency method not implemented.")
