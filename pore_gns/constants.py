"""Units: voxel (2.75 um) and frame (0.25 s); axis order (z, y, x).

Tracers are tracked in the image cropped by MARGIN voxels; the U-Net grid is the uncropped image
subsampled by DS. A tracer at p sits in grid cell (p + MARGIN) // DS.
"""

DS = 8                 # grid cell, voxels (22 um)
MARGIN = (0, 50, 50)   # crop of the tracer frame, voxels
RADIUS = 32            # interaction radius, voxels (88 um)
HISTORY = 5            # velocities fed to the graph network
PATCH = 32 // DS       # phase patch per tracer, cells
