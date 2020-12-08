import glob
import os
import sys


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

from tqdm import tqdm, trange


class Scene:
    CAMERAS = [
        "cam_front",
        "cam_front_left",
        "cam_front_right",
        "cam_back",
        "cam_back_left",
        "cam_back_right",
    ]

    def __init__(self, root, scene: int, args):
        self.root = root
        self.scene = scene
        self.scene_path = self.root / ("scene_%s" % scene)
        self.roi = (args.roi[0] / 2, args.roi[1] / 2)

        # self.compositor = BEVCompositor(resolution=0.04, reach=roi)

        self.grid_map = GridMap(cell_size=args.resolution)

        self.label_img = None
        self.args = args

    def load_sample(self, sample_idx: int, town=None):
        sample_path = self.scene_path / ("sample_%s" % sample_idx)
        if not sample_path.is_dir():
            print("could not find sample folder %s" % sample_path)
            return False

        meta_data = self.__load_sensor_information__(sample_path)
        carla_ego_pose = meta_data["ego_pose"]
        ego_location = carla.Location(
            x=carla_ego_pose["location"][0],  # + 1000,
            y=carla_ego_pose["location"][1],  # + 1000,
            z=carla_ego_pose["location"][2],
        )

        ego_rotation = carla.Rotation(
            roll=carla_ego_pose["rotation"]["roll"],
            pitch=carla_ego_pose["rotation"]["pitch"],
            yaw=carla_ego_pose["rotation"]["yaw"],
        )

        ego_pose = Isometry.from_carla_transform(
            carla.Transform(ego_location, ego_rotation)
        )
        # ego_pose = Isometry.from_matrix(np.asarray(ego_pose.get_matrix()))
        self.ego_pose = ego_pose

        self.compositor = BEVCompositor(resolution=self.args.resolution, reach=self.roi)

        for cam_id in self.CAMERAS:
            camera = meta_data["sensors"][cam_id]
            file_path = PureWindowsPath(camera["data"])
            file_path = Path(file_path.as_posix())

            intrinsic = np.array(camera["intrinsic"])

            # swap rotation yaw = -yaw in extrinisic
            matrix = np.array(camera["extrinsic"])
            """
            rotation = Rotation.from_matrix(matrix[:3, :3])
            r, p, y = rotation.as_euler("xyz")
            rotation = Rotation.from_euler("xyz", (r, -p, -y))
            matrix[:3, :3] = rotation.as_matrix()
            """
            extrinsic = Isometry.from_matrix(matrix)

            cam = BirdsEyeView(id=cam_id, extrinsic=extrinsic, intrinsic=intrinsic)
            self.compositor.addSensor(cam)
            self.compositor.sensors[cam_id].load_data(
                filepath=self.root.joinpath(file_path)
            )
            self.compositor.sensors[cam_id].data = np.fliplr(
                self.compositor.sensors[cam_id].data
            )
            # print("new sample %s" % self.root.joinpath(Path(file_path)))

        # if given a point cloud, use icp to find a rigid transformation instead of the given ego pose
        veh_T_lidar = Isometry.from_matrix(
            np.array(meta_data["sensors"]["lidar_top"]["extrinsic"])
        )
        # swap rotation yaw = -yaw in extrinisic
        matrix = np.array(meta_data["sensors"]["lidar_top"]["extrinsic"])
        rotation = Rotation.from_matrix(matrix[:3, :3])
        r, p, y = rotation.as_euler("xyz")
        rotation = Rotation.from_euler("xyz", (r, p, -y + 180))
        matrix[:3, :3] = rotation.as_matrix()
        veh_T_lidar = Isometry.from_matrix(matrix)

        lidar = Lidar(id="lidar_top", extrinsic=veh_T_lidar)
        file_path = PureWindowsPath(meta_data["sensors"]["lidar_top"]["data"])
        file_path = Path(file_path.as_posix())
        lidar.load_data(filename=self.root.joinpath(file_path.with_suffix(".pcd")))
        lidar.data[1, :] = -lidar.data[1, :]
        lidar.transformLidarToVehicle()

        self.grid_map.update_u(
            point_cloud=lidar.data, veh_T_sensor=veh_T_lidar, world_T_veh=ego_pose
        )

        self.road_boundary_reflected = []
        self.road_boundary = []

        def resolve_invalid_polygons(polygon):
            bb = []
            for b in polygon:
                polyline = shapely.geometry.Polygon(b)
                polyline = polyline.buffer(0)
                if polyline.geom_type == "Polygon":
                    polyline = shapely.geometry.MultiPolygon([polyline])

                for p in polyline.geoms:
                    c = p.exterior.coords[:]
                    bb.append(np.asarray(c))
            return bb

        with (sample_path / "road_polygon.pkl").open("rb") as f:
            road_boundary = pickle.load(f)

            # fix exterior polygon
            # convert both boundaries to nparray
            self.road_boundary.append(resolve_invalid_polygons(road_boundary[0]))
            # self.road_boundary[0] = np.asarray(self.road_boundary[0])
            self.road_boundary.append(resolve_invalid_polygons(road_boundary[1]))

            self.road_boundary_reflected.append(self.road_boundary[0])
            self.road_boundary_reflected.append(self.road_boundary[1])

            for i, c in enumerate(self.road_boundary[0]):
                self.road_boundary[0][i][:, 1] = -c[:, 1]
            for i, c in enumerate(self.road_boundary[1]):
                self.road_boundary[1][i][:, 1] = -c[:, 1]

        self.compositor.render_label(self.road_boundary_reflected)
        self.grid_map.boundaries = self.road_boundary[0]
        self.grid_map.boundaries_interior = self.road_boundary[1]

        self.sample_file = SampleData(
            scene=self.scene,
            sample=sample_idx,
            base_path=self.root / "train_data",
            includes_debug=True,
            prefix=town,
        )

        return True

    def render_sample(self, debug=False):
        img = self.compositor.composeImage(debug=False)
        self.label_img = self.compositor.label

        # [occupation, intensity, height]
        render = self.grid_map.render(debug=False)

        # crop and rotate lidar
        roi_vertices = self.grid_map.get_roi_vertices(roi=self.roi)

        # TODO adopt to new api
        grid = get_rot_bounding_box_experimental(
            render,
            roi_vertices,
            (self.compositor.map_height, self.compositor.map_width),
        )
        # grid = np.flipud(grid)

        direction, angle_map = road_boundary_direction_map(self.label_img)
        inv_map = inverse_distance_map(
            self.label_img,
            truncation_thld=self.args.inverse_distance_thld,
            map_resolution=self.args.resolution,
        )
        end_point = end_point_heat_map(self.label_img)

        self.sample_file.add_data(rgb=img, lidar=grid)
        self.sample_file.add_targets(
            direction_map=direction,
            distance_map=inv_map[:, :, None],
            end_points=end_point[:, :, None],
            ground_truth=self.label_img,
        )

        if debug:
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
            vertices = self.grid_map.get_roi_vertices(roi=self.roi)
            height_bev = get_rot_bounding_box_experimental(
                debug_channels[:, :, 6:9],
                vertices,
                (self.compositor.map_height, self.compositor.map_width),
            )
            # height_bev = np.flipud(height_bev)

            self.sample_file.add_debug_image(name_tag="height_bev", image=height_bev)

            # save labels
            # as the direction map as an angle map for better readability
            normalized_angle_map = np.multiply(
                np.divide(angle_map, 2 * math.pi), 255
            ).astype(np.uint8)
            normalized_angle_map = cv2.applyColorMap(
                normalized_angle_map, cv2.COLORMAP_JET
            )
            normalized_inverse_distance = np.multiply(
                np.divide(inv_map, self.args.inverse_distance_thld),
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

            """
            if self.sample_file.sample % 10 == 0:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(2, 3)
                img /= 255
                ax[0][1].imshow(img)
                ax[0][0].imshow(debug_channels[:, :, :3])

                ax[0][2].set_title("height bev")
                ax[0][2].imshow(height_bev)
                ax[1][0].set_title("inverse_distance_map")
                ax[1][0].imshow(normalized_inverse_distance)
                ax[1][1].set_title("normalized_angle_map")
                ax[1][1].imshow(normalized_angle_map)
                ax[1][2].set_title("normalized_end_points")
                ax[1][2].imshow(normalized_end_points)
                plt.show()
            """
        self.sample_file.write()

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


def write_scene(args, scene_indx, scene_path, includes_debug=False, worker=None):
    root = Path(args.path)
    town_path = root / args.town
    scene_ind = scene_indx

    scene = Scene(town_path, scene=scene_indx, args=args)
    assert str(scene_path) == str(scene.scene_path)

    number_samples = 0
    samples = []
    for i in range(0, 1000):
        sample_dir = scene_path / ("sample_%s" % i)
        if sample_dir.is_dir():
            samples.append(sample_dir)

    for i, sample_dir in enumerate(
        tqdm(
            iterable=samples,
            position=worker,
            desc=("Scene %s" % scene_ind),
            leave=False,
        )
    ):
        if i % args.max_samples_per_grid == 0:
            scene = Scene(town_path, scene=scene_ind, args=args)

        if sample_dir.is_dir():

            if scene.load_sample(i, town=args.town):
                scene.render_sample(debug=includes_debug)


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
                write_scene(
                    args, scene_indx=scene, scene_path=scene_path, includes_debug=debug
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/dominic/data/carla")
    parser.add_argument("--carla_path", type=str, default="/home/dominic/carla")
    parser.add_argument("--town", type=str, default="Town01")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples_per_grid", type=int, default=60)
    parser.add_argument("--roi", nargs="+", type=int, default=(20, 20))
    parser.add_argument("--resolution", type=float, default=0.04)
    parser.add_argument("--inverse_distance_thld", type=float, default=0.5)
    parser.add_argument("--sensor_config", type=str, default="sensor_config.yaml")
    args = parser.parse_args()

    try:
        sys.path.append(
            glob.glob(
                str(
                    Path(args.carla_path)
                    / (
                        "PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
                        % (
                            sys.version_info.major,
                            sys.version_info.minor,
                            "win-amd64" if os.name == "nt" else "linux-x86_64",
                        )
                    )
                )
            )[0]
        )
    except IndexError:
        pass

    import carla

    """
    with open(args.sensor_config, "r") as f:
        sensors = yaml.safe_load(f)
        CAMERAS = sensors["cameras"].keys()
    """
    number_workers = args.workers

    if number_workers == 0:
        number_workers = 1
        main(0)

    else:
        # setup threads
        with ThreadPoolExecutor(max_workers=number_workers) as threads:
            threads.map(main, range(number_workers))
