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

from carla_simulation.helper_classes import Isometry


from pyquaternion import Quaternion
import random
import unittest


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


class carlaTransform(carla.Transform):
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        super(carlaTransform, self).__init__(
            carla.Location(x, y, z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        )


class EquivalencyTest(unittest.TestCase):
    def __init__(self, methodName="runTest", point=[0, 0, 0], transform=Isometry()):
        super(EquivalencyTest, self).__init__(methodName=methodName)

        if isinstance(transform, carla.Transform):
            self.carla_transform = transform
            self.isometry = Isometry.from_carla_transform(self.carla_transform)
        elif isinstance(transform, Isometry):
            self.isometry = transform
            self.carla_transform = self.isometry.to_carla_transform()

        self.point = np.asarray(point)
        self.carla_point = carla.Vector3D(point[0], -point[1], point[2])

    def runTest(self):

        p_ = self.isometry.transform(self.point)
        cp_ = self.carla_transform.transform(self.carla_point)

        msg = (
            "equivalent transforms don't lead to same points for (r,p,y)=(%s,%s,%s)"
            % (
                self.carla_transform.rotation.roll,
                self.carla_transform.rotation.pitch,
                self.carla_transform.rotation.yaw,
            )
        )
        self.assertAlmostEqual(p_[0], cp_.x, places=4, msg=msg)
        self.assertAlmostEqual(p_[1], -cp_.y, places=4, msg=msg)
        self.assertAlmostEqual(p_[2], cp_.z, places=4, msg=msg)


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


# test_unreal_stuff()
test_case_zero_mapping = unittest.TestSuite()
test_case_zero_mapping.addTest(
    EquivalencyTest(point=[0, 0, 0], transform=carlaTransform())
)
test_case_zero_mapping.addTest(EquivalencyTest(point=[0, 0, 0], transform=Isometry()))


test_suite_identity = unittest.TestSuite()
for i in range(10):
    x = random.random()
    y = random.random()
    z = random.random()
    test_suite_identity.addTest(EquivalencyTest(point=[x, y, z], transform=Isometry()))
    test_suite_identity.addTest(
        EquivalencyTest(point=[x, y, z], transform=carlaTransform())
    )

test_suite_yaw = unittest.TestSuite()
test_suite_roll = unittest.TestSuite()
test_suite_pitch = unittest.TestSuite()
for d, i in enumerate(range(-180, 180, 1), 1):
    test_suite_yaw.addTest(
        EquivalencyTest(point=[10, 4, 9], transform=carlaTransform(yaw=i))
    )
    test_suite_roll.addTest(
        EquivalencyTest(point=[10, 4, 9], transform=carlaTransform(roll=i))
    )
    test_suite_pitch.addTest(
        EquivalencyTest(point=[10, 4, 9], transform=carlaTransform(pitch=i))
    )

    test_suite_yaw.addTest(
        EquivalencyTest(
            point=[10, 4, 9],
            transform=Isometry(rotation=Quaternion(axis=[0, 0, 1], degrees=i)),
        )
    )
    test_suite_roll.addTest(
        EquivalencyTest(
            point=[10, 4, 9],
            transform=Isometry(rotation=Quaternion(axis=[1, 0, 0], degrees=i)),
        )
    )
    test_suite_pitch.addTest(
        EquivalencyTest(
            point=[10, 4, 9],
            transform=Isometry(rotation=Quaternion(axis=[0, 1, 0], degrees=i)),
        )
    )

upper_lim = 360
lower_lim = -upper_lim
x = 1
y = 2
z = 3
test_suite_yaw_roll = unittest.TestSuite()
test_suite_roll_yaw = unittest.TestSuite()
test_suite_yaw_pitch = unittest.TestSuite()
test_suite_pitch_yaw = unittest.TestSuite()
test_suite_pitch_roll = unittest.TestSuite()
test_suite_roll_pitch = unittest.TestSuite()
for d, i in enumerate(range(lower_lim, upper_lim, 1), 1):
    conter = range(upper_lim, lower_lim, -1)
    test_suite_yaw_roll.addTest(
        EquivalencyTest(
            point=[x, y, z], transform=carlaTransform(yaw=i, roll=conter[d - 1])
        )
    )
    test_suite_roll_yaw.addTest(
        EquivalencyTest(
            point=[x, y, z], transform=carlaTransform(roll=i, yaw=conter[d - 1])
        )
    )
    test_suite_yaw_pitch.addTest(
        EquivalencyTest(
            point=[10, 4, 9], transform=carlaTransform(yaw=i, pitch=conter[d - 1])
        )
    )
    test_suite_pitch_yaw.addTest(
        EquivalencyTest(
            point=[10, 4, 9], transform=carlaTransform(pitch=i, yaw=conter[d - 1])
        )
    )
    test_suite_pitch_roll.addTest(
        EquivalencyTest(
            point=[10, 4, 9], transform=carlaTransform(pitch=i, roll=conter[d - 1])
        )
    )
    test_suite_roll_pitch.addTest(
        EquivalencyTest(
            point=[10, 4, 9], transform=carlaTransform(roll=i, pitch=conter[d - 1])
        )
    )

test_suite_random = unittest.TestSuite()
for i in range(100):
    roll = random.randrange(lower_lim, upper_lim)
    pitch = random.randrange(lower_lim, upper_lim)
    yaw = random.randrange(lower_lim, upper_lim)
    x = random.randrange(-100, 100)
    y = random.randrange(-100, 100)
    z = random.randrange(-100, 100)
    px = random.randrange(-100, 100)
    py = random.randrange(-100, 100)
    pz = random.randrange(-100, 100)
    test_suite_random.addTest(
        EquivalencyTest(
            point=[px, py, pz],
            transform=carlaTransform(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw),
        )
    )
    """
    test_suite_random.addTest(
        EquivalencyTest(point=[px, py, z], transform=Isometry.random())
    )
    """

main_suite = unittest.TestSuite()
main_suite.addTests(
    [
        test_suite_identity,
        test_case_zero_mapping,
        test_suite_pitch,
        test_suite_roll,
        test_suite_yaw,
        test_suite_pitch_roll,
        test_suite_roll_pitch,
        test_suite_pitch_yaw,
        test_suite_yaw_pitch,
        test_suite_yaw_roll,
        test_suite_roll_yaw,
        test_suite_random,
    ]
)
runner = unittest.TextTestRunner()
runner.run(main_suite)

if __name__ == "__main__":
    main_suite = unittest.TestSuite()
    main_suite.addTests(
        [
            test_suite_identity,
            test_case_zero_mapping,
            test_suite_pitch,
            test_suite_roll,
            test_suite_yaw,
            test_suite_pitch_roll,
            test_suite_roll_pitch,
            test_suite_pitch_yaw,
            test_suite_yaw_pitch,
            test_suite_yaw_roll,
            test_suite_roll_yaw,
            test_suite_random,
        ]
    )
    runner = unittest.TextTestRunner()
    runner.run(main_suite)

    # unittest.main()
