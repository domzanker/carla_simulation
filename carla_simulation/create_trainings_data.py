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

import cv2
import yaml
import pickle
import argparse
import math
import numpy as np

from pathlib import Path, PureWindowsPath
from dataset_utilities.bev_compositor import BEVCompositor
from dataset_utilities.camera import BirdsEyeView, Lidar
from dataset_utilities.transformation import Isometry
from dataset_utilities.grid_map import GridMap
from dataset_utilities.image_utils import (
    road_boundary_direction_map,
    end_point_heat_map,
    inverse_distance_map,
    get_rot_bounding_box_experimental,
)
from dataset_utilities.sample_data import SampleData

from scipy.spatial.transform import Rotation

import shapely

from concurrent.futures import ThreadPoolExecutor
import threading


class Scene:
    CAMERAS = [
        "cam_front",
        "cam_front_left",
        "cam_front_right",
        "cam_back",
        "cam_back_left",
        "cam_back_right",
    ]

    def __init__(self, root, scene: int):
        self.root = root
        self.scene = scene
        self.scene_path = self.root / ("scene_%s" % scene)

        self.compositor = BEVCompositor(resolution=0.04, reach=[15, 10])

        self.grid_map = GridMap(
            cell_size=0.04, sensor_range_u=(60, 60, 0), max_height=0.5
        )

        self.label_img = None

    def load_sample(self, sample_idx: int, town=None):
        sample_path = self.scene_path / ("sample_%s" % sample_idx)
        if not sample_path.is_dir():
            print("could not find sample folder %s" % sample_path)
            return False

        meta_data = self.__load_sensor_information__(sample_path)

        with (sample_path / "road_polygon.pkl").open("rb") as f:
            self.road_boundary = pickle.load(f)

        for cam_id in self.CAMERAS:
            camera = meta_data["sensors"][cam_id]
            file_path = PureWindowsPath(camera["data"])
            file_path = Path(file_path.as_posix())

            intrinsic = np.array(camera["intrinsic"])

            # swap rotation yaw = -yaw in extrinisic
            matrix = np.array(camera["extrinsic"])
            rotation = Rotation.from_matrix(matrix[:3, :3])
            r, p, y = rotation.as_euler("xyz")
            rotation = Rotation.from_euler("xyz", (r, p, -y))
            matrix[:3, :3] = rotation.as_matrix()
            extrinsic = Isometry.from_matrix(matrix)

            cam = BirdsEyeView(id=cam_id, extrinsic=extrinsic, intrinsic=intrinsic)
            cam.load_data(filepath=self.root.joinpath(file_path))
            # print("new sample %s" % self.root.joinpath(Path(file_path)))
            self.compositor.addSensor(cam)

        carla_ego_pose = meta_data["ego_pose"]
        ego_location = carla.Location(*carla_ego_pose["location"])
        ego_rotation = carla.Rotation(
            roll=carla_ego_pose["rotation"]["roll"],
            pitch=carla_ego_pose["rotation"]["roll"],
            yaw=carla_ego_pose["rotation"]["yaw"],
        )
        ego_pose = Isometry.from_carla_transform(
            carla.Transform(ego_location, ego_rotation)
        )

        veh_T_lidar = Isometry.from_matrix(
            np.array(meta_data["sensors"]["lidar_top"]["extrinsic"])
        )
        lidar = Lidar(id="lidar_top", extrinsic=veh_T_lidar)
        file_path = PureWindowsPath(meta_data["sensors"]["lidar_top"]["data"])
        file_path = Path(file_path.as_posix())
        lidar.load_data(filename=self.root.joinpath(file_path.with_suffix(".pcd")))
        self.grid_map.update_u(
            point_cloud=lidar.data, veh_T_sensor=veh_T_lidar, world_T_veh=ego_pose
        )

        self.label_img = cv2.cvtColor(
            cv2.imread(str(sample_path / "road_boundary.png")), cv2.COLOR_BGR2GRAY
        )

        self.label_img = np.fliplr(self.label_img)[1:, 1:, None].astype(np.uint8)
        # self.label_img = np.flipud(self.label_img)

        self.sample_file = SampleData(
            scene=self.scene,
            sample=sample_idx,
            base_path=self.root / "train_data",
            includes_debug=True,
            prefix=town,
        )
        return True

    def render_sample(self):
        img = self.compositor.composeImage(debug=False)

        # [occupation, intensity, height]
        render = self.grid_map.render(debug=False)

        # crop and rotate lidar
        roi_vertices = self.grid_map.get_roi_vertices(roi=(30, 20))
        # TODO adopt to new api
        grid = get_rot_bounding_box_experimental(
            render,
            roi_vertices,
            (self.compositor.map_height, self.compositor.map_width),
        )

        direction, angle_map = road_boundary_direction_map(self.label_img)
        inv_map = inverse_distance_map(self.label_img)
        end_point = end_point_heat_map(self.label_img)

        self.sample_file.add_data(rgb=img, lidar=grid)
        self.sample_file.add_targets(
            direction_map=direction,
            distance_map=inv_map[:, :, None],
            end_points=end_point[:, :, None],
            ground_truth=self.label_img,
        )

        for name, cam in self.compositor.sensors.items():
            if isinstance(cam, BirdsEyeView):
                self.sample_file.add_debug_image(
                    name_tag=("%s" % name),
                    image=cam.data,
                )

        # save debug output of grid
        debug_channels = self.grid_map.render(True)
        self.sample_file.add_debug_image(
            name_tag="grid_occupancy", image=debug_channels[:, :, :3]
        )
        self.sample_file.add_debug_image(
            name_tag="grid_intensity", image=debug_channels[:, :, 3:6]
        )
        self.sample_file.add_debug_image(
            name_tag="grid_height", image=debug_channels[:, :, 6:9]
        )
        self.sample_file.add_debug_image(name_tag="rgb_bev", image=img)
        vertices = self.grid_map.get_roi_vertices(roi=(30, 20))
        height_bev = get_rot_bounding_box_experimental(
            debug_channels[:, :, 6:9],
            vertices,
            (self.compositor.map_height, self.compositor.map_width),
        )

        self.sample_file.add_debug_image(name_tag="height_bev", image=height_bev)

        # save labels
        # as the direction map as an angle map for better readability
        normalized_angle_map = np.multiply(
            np.divide(angle_map, 2 * math.pi), 255
        ).astype(np.uint8)
        normalized_angle_map = cv2.applyColorMap(normalized_angle_map, cv2.COLORMAP_JET)
        normalized_inverse_distance = np.multiply(
            np.divide(inv_map, 5),
            255,
        ).astype(np.uint8)

        normalized_inverse_distance = cv2.applyColorMap(
            normalized_inverse_distance, cv2.COLORMAP_JET
        )

        self.sample_file.add_debug_image(
            name_tag="road_boundary_direction_map",
            image=normalized_angle_map,
        )
        normalized_end_points = np.multiply(end_point, 255).astype(np.uint8)
        normalized_end_points = cv2.applyColorMap(
            normalized_end_points, cv2.COLORMAP_JET
        )
        self.sample_file.add_debug_image(
            name_tag="end_points",
            image=normalized_end_points,
        )
        self.sample_file.add_debug_image(
            name_tag="inverse_distance_map",
            image=normalized_inverse_distance,
        )

        self.sample_file.write()
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3)
        img /= 255
        ax[0][0].imshow(img)
        ax[0][1].imshow(grid[:, :, :3])
        ax[0][2].imshow(self.compositor.label)

        ax[1][0].imshow(angle_map)
        ax[1][1].imshow(inv_map)
        ax[1][2].imshow(end_point)
        plt.show()
        """

    def __load_sensor_information__(self, sample_path):
        yaml_path = sample_path / "sample.yaml"
        with yaml_path.open("r") as f:
            sample_meta = yaml.safe_load(f)
        return sample_meta


def apply_colormap(img, colormap=cv2.COLORMAP_TURBO):
    img = (img + np.pi) / (2 * np.pi) * 255
    img[img > 255] = 255
    if img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1)).astype("uint8")
        batch = [
            cv2.cvtColor(cv2.applyColorMap(img[i], colormap), cv2.COLOR_BGR2RGB)
            for i in range(img.shape[0])
        ]
        img_c = np.stack(batch, axis=0)
        img_c = np.transpose(img_c, (0, 3, 1, 2))
    elif img.ndim == 3:
        img = np.transpose(img, (1, 2, 0)).astype("uint8")
        img_c = np.stack(
            cv2.cvtColor(cv2.applyColorMap(img, colormap), cv2.COLOR_BGR2RGB), axis=0
        )
        img_c = np.transpose(img_c, (2, 0, 1))
    else:
        raise AttributeError
    return img_c


def write_scene(args, scene_indx, scene_path, includes_debug=False):
    root = Path(args.path)
    town_path = root / args.town
    scene_ind = scene_indx

    scene = Scene(town_path, scene=scene_indx, roi=(20, 20))
    assert str(scene_path) == str(scene.scene_path)

    for i in range(5, 1000):
        sample_dir = scene_path / ("sample_%s" % i)

        if i % 60 == 0:
            scene = Scene(town_path, scene=scene_ind, roi=(20, 20))

        if sample_dir.is_dir():
            print(str(sample_dir))

            if scene.load_sample(i, town=args.town):
                scene.render_sample(debug=includes_debug)

    # del scene


def main(worker_index=0):
    global args, number_workers
    # for every scene
    root = Path(args.path)
    town_path = root / args.town
    if worker_index == 0:
        debug = True
    else:
        debug = False
    for scene in range(0, 1000):
        if scene % number_workers == worker_index:
            scene_path = town_path / ("scene_%s" % scene)
            if scene_path.is_dir():
                print(str(scene_path))
                write_scene(
                    args, scene_indx=scene, scene_path=scene_path, includes_debug=debug
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/dominic/data/carla")
    parser.add_argument("--town", type=str, default="Town01")

    parser.add_argument("--sensor_config", type=str, default="sensor_config.yaml")
    args = parser.parse_args()

    """
    with open(args.sensor_config, "r") as f:
        sensors = yaml.safe_load(f)
        CAMERAS = sensors["cameras"].keys()
    """
    number_workers = 3

    if number_workers == 0:
        number_workers = 1
        main(0)

    else:
        # setup threads
        with ThreadPoolExecutor(max_workers=number_workers) as threads:
            threads.map(main, range(number_workers))
