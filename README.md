# Augmented Reality

Augmented Reality (almost) from scratch

## Install

```bash
git clone git@github.com:l4cer/augmented-reality.git
cd augmented-reality
python -m venv venv
./venv/Scripts/activate   # Windows
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

- All models used by default in the code are available on [thingiverse.com](https://www.thingiverse.com/)
- The program has only been tested with .stl files

## Execute

- Connect your camera
- launch the program

```
python main.py [options]
```
- Show to the camera one or more QR codes from the ./images folder (either printed or on your phone) and a 3D model should appear above!

- All options are optional, by default the program displays the vertices of the models stored in the folder 3d_models which match the number of the detected QR code.

```
Options:
  -h                Show this help message and exit
  --debug           Display contours and axes
  --noVertex        Disable vertex rendering
  --polygon         Enable polygon rendering
  --testModel       Display the 1st model before launching the camera
  --path=<path>     Specify a custom path to a 3D model file. 
                    Replaces the first default model in the list.
```

### Examples:

```
python main.py --debug --polygon --path="3d_models/my_model.stl"
python main.py --testModel --noVertex
```