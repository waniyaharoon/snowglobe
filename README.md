# Snow Globe

A tiny software 3D renderer written in Python with NumPy, displayed with pygame.
Every object is a cloud of points with surface normals, lit with Lambert shading,
depth-sorted with a z-buffer, and drawn as coloured ASCII-style cells.

## Run

    pip install numpy pygame-ce
    python snowglobe.py

Press ESC to quit.

## What it demonstrates

- Point-cloud shapes: sphere, tube, torus, surface of revolution
- Perspective projection and a moving camera with easing
- Lambert lighting with moon, lamp and ambient light
- Z-buffer for hidden surfaces
- Alpha-blended glass dome with a Fresnel-style rim
- ASCII-style rendering from a glyph atlas