import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
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
from dataset import Dataset
from shapely import speedups

if speedups.available:
    speedups.enable()
    print("Speedups enabled")

from pathlib import Path
import yaml
from write_scene import write_scene

from tqdm import tqdm, trange


def available_spawn_points(args):
    # create a world
    # first define a client
    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(10.0)  # seconds

    town = args.map
    step_delta = args.step_delta
    duration = args.scene_length

    world = client.load_world(town)
    spawn_points = world.get_map().get_spawn_points()
    return spawn_points


def main(args):

    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(10.0)  # seconds

    world = client.load_world(args.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    if args.spawn_point >= 0:
        write_scene(args, client=client, world=world)
    else:
        spawn_points = available_spawn_points(args)
        with tqdm(total=args.number_of_scenes) as pbar:
            for i, spawn_point in tqdm(enumerate(spawn_points)):
                if i >= args.number_of_scenes:
                    break
                pbar.set_description(
                    "[%s] write scene: %s / %s"
                    % (args.map, i, args.number_of_scenes - 1)
                )
                pbar.update(i + 1)
                args.spawn_point = i + 1
                write_scene(args, client=client, world=world)
                client.reload_world()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--server_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=2000)
    parser.add_argument("--base_path", type=str, default="/tmp/carla")
    parser.add_argument("--carla_path", type=str, default="/home/dominic/carla")

    parser.add_argument("--step_delta", type=float, default=0.05)
    parser.add_argument("--scene_length", type=int, default=90)

    parser.add_argument("--spawn_point", type=int, default=-1)
    parser.add_argument("--number_of_scenes", type=int, default=45)

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

    for town in ["Town03", "Town07", "Town10"]:
        args.map = town
        args.spawn_point = -1

        main(args)
