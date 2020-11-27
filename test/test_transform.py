import glob
import os
import sys
from math import sin, cos, radians, degrees

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
import numpy as np

from dataset_utilities.transformation import Isometry


from pyquaternion import Quaternion
import random


def print_report(args):
    print("")
    print("")
    print("======== REPORT ========")
    print("==")
    print("==    failed in dim %s " % args[0])
    print("==    points %s <==> %s " % (args[1], args[2]))
    print("==    roll: %s, pitch: %s, yaw: %s " % (args[3], args[4], args[5]))
    print("==    x: %s, y: %s, z: %s " % (args[6], args[7], args[8]))
    print("==")
    print("========================")
    print("")
    print("")


def test_equivalent(name, point, carla_transform: carla.Transform, nmbr=1):
    print("[%s][%s] test equivalent mapping..." % (name, nmbr), end="")

    isometry = Isometry.from_carla_transform(carla_transform)

    carla_point = carla.Vector3D(point[0], -point[1], point[2])
    point = np.asarray(point)

    p_ = isometry.transform(point)
    cp_ = carla_transform.transform(carla_point)

    tol = 1e-3
    try:
        np.testing.assert_allclose(p_, np.array([cp_.x, -cp_.y, cp_.z]), rtol=tol)
    except AssertionError as e:
        print_report(
            (
                0,
                cp_,
                p_,
                carla_transform.rotation.roll,
                carla_transform.rotation.pitch,
                carla_transform.rotation.yaw,
                carla_transform.location.x,
                carla_transform.location.y,
                carla_transform.location.z,
            )
        )
        raise e
    print("passed")


class carlaTransform(carla.Transform):
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        super(carlaTransform, self).__init__(
            carla.Location(x, y, z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        )


def test_unreal_stuff():
    roll = 0
    pitch = 90
    yaw = -90

    transform = carlaTransform(roll=roll, pitch=pitch, yaw=yaw)

    cy = cos(radians(yaw))
    sy = sin(radians(yaw))

    cp = cos(radians(pitch))
    sp = sin(radians(pitch))

    cr = cos(radians(roll))
    sr = sin(radians(roll))

    m = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, 0],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, 0],
            [sp, -cp * sr, cp * cr, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    print(m)
    m1 = np.array(
        [
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr],
        ]
    )
    m2 = np.array(
        [
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ]
    )
    m3 = np.array(
        [
            [cy, sy, 0],
            [-sy, cy, 0],
            [0, 0, 1],
        ]
    )
    m_ = m3 @ m2 @ m1
    cm = np.array(transform.get_matrix())
    print(cm)
    np.testing.assert_allclose(m_, m[:3, :3])
    np.testing.assert_allclose(m, cm, rtol=1e-5, atol=10)


if __name__ == "__main__":

    # test_unreal_stuff()

    identity_transform = carlaTransform()

    test_equivalent("zero mapping", [0, 0, 0], identity_transform)
    for i in range(10):
        x = random.random()
        y = random.random()
        z = random.random()
        test_equivalent("identity mapping", [x, y, z], identity_transform, i)

    for d, i in enumerate(range(-180, 180, 1), 1):
        test_equivalent("yaw", [10, 4, 9], carlaTransform(yaw=i), d)
        test_equivalent("roll", [10, 4, 9], carlaTransform(roll=i), d)
        test_equivalent("pitch", [10, 4, 9], carlaTransform(pitch=i), d)

    upper_lim = 360
    lower_lim = -upper_lim
    x = 1
    y = 2
    z = 3
    for d, i in enumerate(range(lower_lim, upper_lim, 1), 1):
        conter = range(upper_lim, lower_lim, -1)
        test_equivalent(
            "yaw-roll", [x, y, z], carlaTransform(yaw=i, roll=conter[d - 1]), d
        )
        test_equivalent(
            "roll-yaw", [x, y, z], carlaTransform(roll=i, yaw=conter[d - 1]), d
        )

    for d, i in enumerate(range(lower_lim, upper_lim, 1), 1):
        conter = range(upper_lim, lower_lim, -1)
        test_equivalent(
            "yaw-pitch", [10, 4, 9], carlaTransform(yaw=i, pitch=conter[d - 1]), d
        )
        test_equivalent(
            "pitch-yaw", [10, 4, 9], carlaTransform(pitch=i, yaw=conter[d - 1]), d
        )

    for d, i in enumerate(range(lower_lim, upper_lim, 1), 1):
        conter = range(upper_lim, lower_lim, -1)
        test_equivalent(
            "pitch-roll", [10, 4, 9], carlaTransform(pitch=i, roll=conter[d - 1]), d
        )
        test_equivalent(
            "roll-pitch", [10, 4, 9], carlaTransform(roll=i, pitch=conter[d - 1]), d
        )

    for d, i in enumerate(range(100), 1):
        roll = random.randrange(lower_lim, upper_lim)
        pitch = random.randrange(lower_lim, upper_lim)
        yaw = random.randrange(lower_lim, upper_lim)
        x = random.randrange(-100, 100)
        y = random.randrange(-100, 100)
        z = random.randrange(-100, 100)
        px = random.randrange(-100, 100)
        py = random.randrange(-100, 100)
        pz = random.randrange(-100, 100)
        test_equivalent(
            "free",
            [px, py, pz],
            carlaTransform(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw),
            d,
        )
