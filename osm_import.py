import glob
import os
import sys

import argparse
from pathlib import Path


def main(args):
    path = Path(args.path)
    with path.open("r") as f:
        osm_data = f.read()

    settings = carla.Osm2OdrSettings()
    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    with path.with_suffix(".xodr").open("w+") as f:
        f.write(xodr_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla_path", type=str, default="/home/dominic/carla")
    parser.add_argument("--path", type=str, default="/home/dominic/data/carla")
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

    main(args)
