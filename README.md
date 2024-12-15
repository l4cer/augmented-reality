# Augmented Reality

Augmented Reality (almost) from scratch

## Install

To clone the project and install the dependencies, run the following commands:

```bash
git clone git@github.com:l4cer/augmented-reality.git
cd augmented-reality
python -m venv venv
./venv/Scripts/activate   # Windows
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

## Execute

To run the program, connect your camera and execute the following command:

```
python main.py [options]
```

### Usage

Show to the camera one or more ArUco from the `markers` folder (these can be either printed or displayed on a screen, such as your phone). A 3D model should appear over the detected markers!

By default, the program uses the first video capture device (device 0). If needed, you can customize the behavior using the options below.

### Options

```
-h                 Display this help message and exit.
--debug            Enable debug mode to visualize contours and axes.
--device=<int>     Specify the video capture device index.
```

⚠️ **Note:** if you experience issues with the default camera device, use the `--device` option to specify a different camera index.

### Examples

```
python main.py
python main.py --debug
python main.py --device=0
```
