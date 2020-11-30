import logging
import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from typing import List, Tuple

from shapely.geometry import Polygon, MultiPolygon

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "/home/dominic/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla


def to_homogenous_points(points: np.ndarray):

    try:
        shape = points.shape
    except AttributeError:
        logging.warn(
            "points should allways be provided as numpy arrays not %s" % type(points)
        )
        points.append(1)
        return points

    if points.ndim == 1:
        points = np.expand_dims(points, axis=-1)
        shape = points.shape

    assert shape[0] in [2, 3]
    assert points.ndim == 2

    ones = np.broadcast_to(np.ones(1, dtype=points.dtype), (1, shape[1]))
    return np.vstack((points, ones))


class Isometry:

    """
    Implements a Class for isometric transformations

    The transformation is composed as T * R

    This class can import and export carla.Transforms
    """

    def __init__(
        self,
        translation: np.ndarray = np.zeros(3),
        rotation: Quaternion = Quaternion(),
    ):
        self.translation = np.asarray(translation)
        self.rotation = Quaternion(rotation)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        # TODO assert validity of matrix
        return cls(matrix[:3, 3], Quaternion(matrix=matrix))

    @classmethod
    def from_carla_transform(cls, trafo):
        matrix = np.asarray(trafo.get_matrix())
        # 1. convert the translation to right-handed coordinate system
        rh_translation = np.array([matrix[0, 3], -matrix[1, 3], matrix[2, 3]])

        # 2. convert the rotation to right-handed coordinate system
        yaw = trafo.rotation.yaw
        pitch = trafo.rotation.pitch
        roll = trafo.rotation.roll
        # apply rotation (pitch->yaw->roll)
        # build from euler
        # carla documentation states rotation by YZX
        # https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation
        rotation = Rotation.from_euler("xyz", (roll, -pitch, -yaw), degrees=True)
        (x, y, z, w) = rotation.as_quat()
        rh_quaternion = Quaternion(w, x, y, z)
        return cls(rh_translation, rh_quaternion)

    def to_carla_transform(self):
        location = carla.Location(
            x=self.translation[0], y=self.translation[1], z=self.translation[2]
        )
        (w, x, y, z) = self.rotation.normalised.elements
        rotation = Rotation.from_quat((x, y, z, w))
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
        rot = carla.Rotation(roll=roll, pitch=-pitch, yaw=-yaw)
        return carla.Transform(location, rot)

    def translate(self, other: np.ndarray) -> np.ndarray:
        assert other.shape[-1] == 3
        return other + self.translation

    def rotate(self, other: np.ndarray) -> np.ndarray:
        return self.rotate.rotate(other)

    def inverse(self):
        """Return the inverse Isometry"""
        return self.__invert__()

    def invert(self) -> None:
        """Invert the Isometry Object"""
        inv_ = ~self
        self.translation = inv_.translation
        self.rotation = inv_.rotation

    @property
    def matrix(self) -> np.ndarray:
        return self.translation_matrix @ self.rotation_matrix

    # compatibility function when working with carla.Transform
    def get_matrix(self) -> List[List[float]]:
        return self.matrix.tolist()

    # compatibility function when working with carla.Transform
    def get_inverse_matrix(self) -> List[List[float]]:
        return self.inverse().matrix.tolist()

    @matrix.setter
    def matrix(self, matrix):
        t = self.from_matrix(matrix)
        self.translation = t.translation
        self.rotation = t.rotation

    @property
    def rotation_matrix(self):
        return self.rotation.transformation_matrix

    @property
    def translation_matrix(self):
        t = np.array(
            [
                [1, 0, 0, self.translation[0]],
                [0, 1, 0, self.translation[1]],
                [0, 0, 1, self.translation[2]],
                [0, 0, 0, 1],
            ]
        )
        return t

    def transform(self, other) -> np.ndarray:
        """Apply Isometry to other"""
        return self @ other

    @classmethod
    def random(cls):
        """Return a random Isometry"""
        random_rotation = Quaternion.random()
        random_translation = np.random.random(3)
        return cls(translation=random_translation, rotation=random_rotation)

    def __mul__(self, other):
        return self @ other

    def __matmul__(self, other):
        if isinstance(other, Isometry):
            return Isometry.from_matrix(self.matrix @ other.matrix)
        elif isinstance(other, (np.ndarray, list, tuple)):
            other = to_homogenous_points(other)
            prod = self.matrix @ other
            return np.squeeze(prod[:-1] / prod[-1])

    def __invert__(self):
        M_inv = np.eye(4)
        M_inv[:3, :3] = self.rotation.rotation_matrix.T
        M_inv[:3, 3] = -(M_inv[:3, :3] @ self.translation)
        return Isometry.from_matrix(M_inv)

    def __eq__(self, other):
        if isinstance(other, Isometry):
            return (
                self.rotation == other.rotation
                and (self.translation == other.translation).all()
            )
        elif isinstance(other, np.ndarray):
            if other.shape != (4, 4):
                raise ValueError(
                    "Expected Other of shape (4, 4), got %s" % (other.shape)
                )
            return np.array_allclose(self.matrix == other)
        else:
            raise TypeError(
                "Expected Other of type Isometry or numpy.ndarray, got %s" % type(other)
            )

    def __str__(self):
        return str(self.matrix)


class Transformation(Isometry):
    """
    This class just serves as an alias for Isometry at this point
    """

    def __init__(
        self,
        translation: np.ndarray = np.zeros(3),
        rotation: Quaternion = Quaternion(),
    ):
        super().__init__(translation, rotation)


class RoadBoundary:
    """
    RoadBoundary described a PolyLine segment in vehicle coordinates
    """

    def __init__(self, polyline: List[Tuple[float, float]] = []):
        self.boundary = polyline

    @classmethod
    def from_shapely(cls, shapely_polygon):
        if isinstance(shapely_polygon, Polygon):
            polygons = [shapely_polygon]
        elif isinstance(shapely_polygon, MultiPolygon):
            polygons = shapely_polygon.geoms
        else:
            raise ValueError

        # unpack all points in
        road_boundary = []
        for polygon in polygons:
            exterior = [xy for xy in polygon.exterior.coords]

            interiors = []
            for interior in polygon.interiors:
                interiors.append([xy for xy in polygon.interiors.coords])

    def as_image(self, image_dims: Tuple[int, int]):

        raise NotImplementedError

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    pass
