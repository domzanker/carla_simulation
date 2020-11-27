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

from dataset_utilities.camera import BirdsEyeView
from dataset_utilities.transformation import Isometry

# first define a client
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)  # seconds

world = client.load_world("Town03")
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


class SensorPlatform:
    def __init__(self, world):
        self.world = world

        self.vehicle = world.get_blueprint_library().find("vehicle.mercedes-benz.coupe")
        self.vehicle.set_attribute("role_name", "ego")
        self.spawn_points = world.get_map().get_spawn_points()

        self.ego_pose = self.spawn_points[2]
        self.ego_vehicle = world.spawn_actor(self.vehicle, self.ego_pose)
        self.ego_vehicle.set_autopilot(True)

        """
        self.CAM_FRONT = self.add_camera(carla.Transform(carla.Location(x=1, z=2)))
        self.CAM_FRONT_RIGHT = self.add_camera(
            carla.Transform(carla.Location(x=0.5, y=-1, z=2), carla.Rotation(yaw=-45))
        )
        self.CAM_FRONT_LEFT = self.add_camera(
            carla.Transform(carla.Location(x=0.5, y=1, z=2), carla.Rotation(yaw=45))
        )
        self.CAM_BACK_RIGHT = self.add_camera(
            carla.Transform(carla.Location(x=-0.5, y=-1, z=2), carla.Rotation(yaw=-135))
        )
        self.CAM_BACK_LEFT = self.add_camera(
            carla.Transform(carla.Location(x=-0.5, y=1, z=2), carla.Rotation(yaw=135))
        )
        self.CAM_BACK = self.add_camera(
            carla.Transform(carla.Location(x=-1, z=2), carla.Rotation(yaw=180))
        )
        """
        self.cameras = {}

        self.LIDAR_TOP = None

    def add_camera(
        self,
        name,
        transform=carla.Transform(),
        image_size_x=1920,
        image_size_y=1080,
        blueprint="sensor.camera.rgb",
        **kwargs,
    ):
        blueprint = world.get_blueprint_library().find(blueprint)
        blueprint.set_attribute("image_size_x", str(image_size_x))
        blueprint.set_attribute("image_size_y", str(image_size_y))
        blueprint.set_attribute("sensor_tick", "1.0")
        for key, val in kwargs.items():
            blueprint.set_attribute(key, str(val))
        # Set the time in seconds between sensor captures
        K = self._get_K(blueprint)
        image_height = image_size_y
        focal_dist = K[0, 0]

        z = focal_dist * (20 / (0.5 * image_height))

        extrinsic = carla.Transform(carla.Location(z=z), transform.rotation)
        sensor = self.world.spawn_actor(
            blueprint, extrinsic, attach_to=self.ego_vehicle
        )
        q_ = Queue(1)
        self.cameras[name] = (sensor, q_)
        # sensor.listen(lambda data: self.reference_callback(data, q_))
        sensor.listen(lambda data: q_.put(data))

        bev = BirdsEyeView(
            name,
            extrinsic=Isometry.from_carla_transform(extrinsic),
            intrinsic=K,
            resolution=0.02,
            offset=(10, 10),
            out_size=(1000, 2000),
        )
        return bev, q_

    def reference_callback(self, data, queue):
        self.ego_pose = self.ego_vehicle.get_transform()
        queue.put(data)

    def _get_K(self, camera_bp):

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
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


if __name__ == "__main__":

    platform = SensorPlatform(world)

    spectator = world.get_spectator()
    spectator.set_transform(platform.CAM_FRONT.get_transform())

    while True:
        world.tick()

    print(platform.ego_vehicle.transform)
    # record
    client.start_recorder("test_recording.log")
