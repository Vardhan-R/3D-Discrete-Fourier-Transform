# 3d-Discrete-Fourier-Transform
Discrete Fourier transform in 3d using PyOpenGL.

## Installation
1. Download the repository (or just the [`main.py`](main.py) file will suffice).

2. Make sure that [`PyOpenGL`](https://pyopengl.sourceforge.net/) is installed (you may use `pip` to install it: `$ pip install PyOpenGL PyOpenGL_accelerate`).

## Usage
1. Run the `main.py` file.

2. You may change the window width and height on line 7 of the file.

3. Below that are various examples for starting points, including randomly generated points, preset points and a helix.

4. The variable `dt` can be varied to change the drawing speed, and the variable `cyclic_drawing` can be set to `False` to _not_ join the last point to the first.

## Theory
Click [here](docs/Multidimensional_Discrete_Fourier_Transform.pdf) to know more about the mathematical details.
