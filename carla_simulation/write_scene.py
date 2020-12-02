import glob
import os
import sys
import cv2

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
from dataset import Dataset
from shapely import speedups

if speedups.available:
    speedups.enable()
    print("Speedups enabled")

from pathlib import Path
import yaml
import pickle


def write_scene(args, client=None, world=None):

    if client is None:
        # create a world
        # first define a client
        client = carla.Client(args.server_addr, args.server_port)
        client.set_timeout(10.0)  # seconds

    if world is None:
        world = client.load_world(town)
        world.set_weather(carla.WeatherParameters.ClearNoon)

    town = args.map
    step_delta = args.step_delta
    duration = args.scene_length

    spectator = world.get_spectator()

    # TODO: setup folder structure
    base = Path(args.base_path)

    root = base / town
    root.mkdir(parents=True, exist_ok=True)

    spawn_point = args.spawn_point
    scene_dir = root / ("scene_%s" % spawn_point)

    ticks_per_scene = int(duration // step_delta)
    ticks_per_second = 1 / step_delta
    # fix sensor rate at 2hz
    sample_rate = 2  # [hz]

    ticks_per_sample = ticks_per_second / sample_rate

    dataset = Dataset(
        world, vehicle_spawn_point=spawn_point, sensor_tick=1 / sample_rate
    )

    # TODO tidy up
    tm = client.get_trafficmanager(8000)
    # dataset.sensor_platform.ego_vehicle.set_target_velocity(carla.Vector3D(50, 0, 0))
    dataset.sensor_platform.ego_vehicle.set_autopilot(True, 8000)

    spec = dataset.ego_pose
    spec.location.z = 3
    spectator.set_transform(spec)
    step = 0

    settings = world.get_settings()
    settings.synchronous_mode = True  # True
    settings.fixed_delta_seconds = step_delta
    world.apply_settings(settings)

    [world.tick() for _ in range(5)]

    for i in range(ticks_per_scene):

        frame = world.tick()
        print("simulation time: frame %s -- %s sec" % (frame, round(i * step_delta, 3)))
        if i % ticks_per_sample == 0:  # and i > 0:
            print("sensor tick")
            sample_dir = scene_dir / ("sample_%s" % step)
            sample_dir.mkdir(parents=True, exist_ok=True)
            # generate one sample from every tick

            sample = dataset.get_sample(frame_id=frame - 5, include_map=True)
            if sample == False:
                continue

            spec = dataset.ego_pose
            spec.location.z = 2
            spectator.set_transform(spec)

            sample_dict = {}
            cv2.imwrite(str(sample_dir / "road_boundary.png"), dataset.boundaries_img)

            sample_dict["ego_pose"] = {
                "rotation": {
                    "roll": dataset.ego_pose.rotation.roll,
                    "pitch": dataset.ego_pose.rotation.roll,
                    "yaw": dataset.ego_pose.rotation.yaw,
                },
                "location": [
                    dataset.ego_pose.location.x,
                    dataset.ego_pose.location.y,
                    dataset.ego_pose.location.z,
                ],
            }
            sample_dict["sensors"] = {}
            for name, lidar in dataset.lidars.items():
                export_dict = {}
                export_file = lidar.exportPCD(sample_dir)
                export_dict["data"] = str(
                    export_file.relative_to(root).with_suffix(".pcd")
                )
                export_dict["extrinsic"] = lidar.M.tolist()
                sample_dict["sensors"][lidar.id] = export_dict

            for name, cam in dataset.cameras.items():
                export_dict = {}
                export_file = cam.write_data(sample_dir)
                export_dict["data"] = str(export_file.relative_to(root))
                export_dict["extrinsic"] = cam.M.tolist()
                export_dict["intrinsic"] = cam.K.tolist()
                sample_dict["sensors"][cam.id] = export_dict

            boundary_file = sample_dir / "road_polygon.pkl"
            with boundary_file.open("wb+") as f:
                pickle.dump(dataset.road_boundaries, f)

            sample_file = sample_dir / "sample.yaml"
            with sample_file.open("w+") as f:
                yaml.safe_dump(sample_dict, f)
            del sample_dict
            step += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--server_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=2000)
    parser.add_argument("--base_path", type=str, default="/home/dominic/data/carla")

    parser.add_argument("--step_delta", type=float, default=0.05)
    parser.add_argument("--scene_length", type=int, default=90)

    parser.add_argument("--spawn_point", type=int, default=0)

    args = parser.parse_args()
    write_scene(args)
