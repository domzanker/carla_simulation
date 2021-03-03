
# CARLA_SIMULATION
This repo includes code that is needed in order to create simulated datasets from the CARLA simulation environment
## Installation/ Requirements
#### CARLA source
Note that Release 0.9.10 causes crashes in the camera sensors, use version 0.9.10.1 or later.  
In order to find the CARLA-PythonAPI symlink the carla directory into this repo.  
`ln -s /path/to/carla carla`  
Alternatively you can change the path in the code.
from carla_simulation/carla_simulation
You will have to update this accordingly in the code.

#### Python setup
By now Carla only runs on Python3.7
I recommend creating a virtualenvironment.
Install dependecies by running  
`pip install -r requirements.txt`  
## Usage
This repository includes several skripts for creating simulated training data: 

#### write_map.py
This script collects data from a given map in carla for tracks of a given length.
The collected data is than write to disk.

First start the simulation server (assuming carla symlink)  
`./carla/CarlaUE4.sh -opengl`

then run  
`python write_map.py --map TOWNXX (additional args)`  
use the `-h` flag for available arguments

There are some known issues. Most noticable are missing sensor data while recording. 
This seems to be an issue with carla as it appears more often on some maps.


#### create_trainings_data.py
After writing data from the simulation, create h5 files for training by running  
`python create_trainings_data.py --path /path/to/simulated/data (additional args)`
use the `-h` flag for available arguments


