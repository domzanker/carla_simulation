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
from queue import Queue
import numpy as np
from scipy.spatial.transform import Rotation

from dataset_utilities.camera import BirdsEyeView, Camera, Lidar
from dataset_utilities.transformation import Isometry

from pyquaternion import Quaternion


class SensorPlatform:
    def __init__(self, world, spawn_point):
        self.world = world

        self.vehicle = self.world.get_blueprint_library().find(
            # "vehicle.mercedes-benz.coupe"
            "vehicle.bmw.isetta"
        )
        self.vehicle.set_attribute("role_name", "ego")

        self.ego_vehicle = world.spawn_actor(self.vehicle, spawn_point)
        self.ego_vehicle.set_autopilot(True)

        self.cameras = {}
        self.lidars = {}

        imu_bp = self.world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "1.0")
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.ego_pose = Queue()
        self.imu.listen(lambda data: self.ego_pose.put(data))

    def add_topview(
        self,
        name,
        veh_T_sensor=carla.Location(),  # [m]
        resolution=0.02,  # [m/px]
        roi=[20, 15],  # [m]
        blueprint="sensor.camera.rgb",
        **kwargs,
    ):
        blueprint = self.world.get_blueprint_library().find(blueprint)
        # Set the time in seconds between sensor captures
        blueprint.set_attribute("sensor_tick", "1.0")

        image_size_x = int(roi[1] // resolution)
        image_size_y = int(roi[0] // resolution)

        blueprint.set_attribute("image_size_x", str(image_size_x))
        blueprint.set_attribute("image_size_y", str(image_size_y))
        for key, val in kwargs.items():
            blueprint.set_attribute(key, str(val))

        K = self._get_K(blueprint)
        image_height = image_size_y
        focal_dist = K[0, 0]

        z = focal_dist * (roi[0] / image_height)

        veh_T_sensor.location.z = z
        sensor = self.world.spawn_actor(
            blueprint, veh_T_sensor, attach_to=self.ego_vehicle
        )
        sensor.set_location(veh_T_sensor.location)
        q_ = Queue()
        self.cameras[name] = (sensor, q_)
        # sensor.listen(lambda data: self.reference_callback(data, q_))
        sensor.listen(lambda data: q_.put(data))

        bev = Camera(
            name,
            # extrinsic=Isometry.from_carla_transform(veh_T_sensor),
            extrinsic=build_extrinsic(veh_T_sensor),
            intrinsic=K,
            crop_horizon=False,
        )
        return bev, q_, sensor.get_transform()

    def add_camera(
        self,
        name,
        veh_T_sensor=carla.Transform(),
        K=None,
        blueprint="sensor.camera.rgb",
        **kwargs,
    ):
        blueprint = self.world.get_blueprint_library().find(blueprint)
        blueprint.set_attribute("sensor_tick", "1.0")
        for key, val in kwargs.items():
            blueprint.set_attribute(key, str(val))
        # Set the time in seconds between sensor captures
        if K is not None:
            raise NotImplementedError
        else:
            K = self._get_K(blueprint)

        sensor = self.world.spawn_actor(
            blueprint, veh_T_sensor, attach_to=self.ego_vehicle
        )
        sensor.set_location(veh_T_sensor.location)

        q_ = Queue()
        self.cameras[name] = (sensor, q_)
        # sensor.listen(lambda data: self.reference_callback(data, q_))
        sensor.listen(lambda data: q_.put(data))

        extrinsic = build_extrinsic(veh_T_sensor)
        bev = BirdsEyeView(
            name,
            extrinsic=extrinsic,
            intrinsic=K,
            resolution=0.04,
            offset=(10, 0),
            out_size=(500, 750),
            crop_horizon=True,
        )
        return bev, q_, sensor.get_transform()

    def add_lidar(
        self,
        name,
        veh_T_sensor=carla.Transform(),
        blueprint="sensor.lidar.ray_cast",
        **kwargs,
    ):
        blueprint = self.world.get_blueprint_library().find(blueprint)
        for key, val in kwargs.items():
            blueprint.set_attribute(key, str(val))
        sensor = self.world.spawn_actor(
            blueprint, veh_T_sensor, attach_to=self.ego_vehicle
        )
        blueprint.set_attribute("sensor_tick", "1.0")
        q_ = Queue()
        self.lidars[name] = (sensor, q_)
        sensor.listen(lambda data: q_.put(data))
        lidar = Lidar(id=name, extrinsic=Isometry.from_carla_transform(veh_T_sensor))
        return lidar, q_, sensor.get_transform()

    def reference_callback(self, data, queue):
        self.ego_pose = self.ego_vehicle.get_transform()
        queue.put(data)

    def _from_K(self, K):
        pass
        # image_width = 2*K[]

    def _get_K(self, camera_bp):

        # Build the K projection matrix:
        # K = [[mx,  0, image_w/2],
        #      [ 0, my, image_h/2],
        #      [ 0,  0,         1]]
        # with m{i} = f*m{i}
        #       f = focal length in [m]
        #       m{i}= inverse px size [px/m]

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal_x = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = focal_x
        K[1, 1] = focal_x
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        return K


def build_extrinsic(trafo):
    # carla applies rotation first, then translates.
    # for a intuitive declaration we have to convert this
    """
    yaw = trafo.rotation.yaw
    pitch = trafo.rotation.pitch
    roll = trafo.rotation.roll
    # apply rotation (pitch->yaw->roll)
    # build from euler
    # carla documentation states rotation by YZX
    # https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation
    rotation = Rotation.from_euler("xyz", (roll, -pitch, -yaw), degrees=True)
    origin = rotation.apply(
        np.array([trafo.location.x, -trafo.location.y, trafo.location.z]), inverse=False
    )
    trafo.location = carla.Location(origin[0], origin[1], origin[2])
    """

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    matrix = Isometry.from_carla_transform(trafo)

    base_change = Rotation.from_euler("XY", (90, -90), degrees=True)
    (x, y, z, w) = base_change.as_quat()
    base_change = Isometry(rotation=Quaternion(w, x, y, z))

    return matrix @ base_change


if __name__ == "__main__":
    # first define a client
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)  # seconds

    world = client.load_world("Town03")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    platform = SensorPlatform(world)

    spectator = world.get_spectator()
    spectator.set_transform(platform.CAM_FRONT.get_transform())

    while True:
        world.tick()

    print(platform.ego_vehicle.transform)
    # record
    client.start_recorder("test_recording.log")
