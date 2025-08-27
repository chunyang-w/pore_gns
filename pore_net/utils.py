# Code from Chunyang Wang https://github.com/chunyang-w & Yuxuan Gu https://github.com/guyuxuan9
import torch
import pyvista as pv
import numpy as np
import torch.nn.functional as F
import numba
import glob
from natsort import natsorted
import tifffile

# import pygmsh
from skimage import measure
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


def get_stats(data_list):
    """
    Get statistics of the data - for normalization and denormalization.
    """
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    mean_vec_y = torch.zeros(data_list[0].y.shape[2:])
    std_vec_y = torch.zeros(data_list[0].y.shape[2:])

    eps = torch.tensor(1e-8)

    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    for dp in data_list:

        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x**2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr**2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y[0, :, :], dim=0)
        std_vec_y += torch.sum(dp.y[0, :, :] ** 2, dim=0)
        num_accs_y += dp.y.shape[1]

    mean_vec_x = mean_vec_x / num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accs_x - mean_vec_x**2), eps)

    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(
        torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge**2),
        eps,
    )

    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ]

    return mean_std_list


def get_dice_score(pred, gt):
    return (2.0 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)


def load_tif(t_idx, tif_dir, pore_mask, exp_id, ds_patch):
    tif_files = glob.glob(f"{tif_dir}/*.tif")
    tif_files = natsorted(tif_files)
    # print(tif_path, t_idx, exp_id)
    if exp_id == 72:
        t_idx = int(t_idx // 2)
        tif_path = tif_files[t_idx]
        tif_file = tifffile.imread(tif_path)[:, 50:-50, 50:-50]
        tif_data = torch.zeros(1228, 1466, 1466, dtype=torch.short)
        tif_data[510:, :, :] = torch.tensor(tif_file // 255, dtype=torch.short)
        tif_data = down_sample(tif_data, ds_patch)
    elif exp_id == 73:
        tif_path = tif_files[t_idx]
        tif_data = tifffile.imread(tif_path)[:, 50:-50, 50:-50]
        tif_data = down_sample(tif_data, ds_patch)
        tif_data = torch.tensor(tif_data, dtype=torch.short)  # noqa E501
    else:
        raise ValueError(f"Experiment ID {exp_id} not supported")
    tif_data[pore_mask] = -1
    return tif_data


def extract_patches(positions, tif_data, patch_size_raw=32, ds_patch=1, ds=1):
    """
    Modified function that extracts patches in a fully vectorized manner
    on GPU.
    """
    tif_data = tif_data.to(positions.device)

    # Define patch sizes
    # patch_size = patch_size_raw // 2  # 16//2 = 8
    patch_size = patch_size_raw // ds_patch

    half_patch = patch_size // 2  # 8//2 = 4

    # Convert raw coordinates to downsampled coordinates with offset
    # oil_nodes_downsampled = (positions // 2).int()
    oil_nodes_downsampled = (positions // ds_patch).int()

    # Compute valid patch starting indices along each dimension
    x_min = torch.clamp(
        oil_nodes_downsampled[:, 0] - half_patch,
        0,
        tif_data.shape[0] - patch_size,  # noqa E501
    )
    y_min = torch.clamp(
        oil_nodes_downsampled[:, 1] - half_patch,
        0,
        tif_data.shape[1] - patch_size,  # noqa E501
    )
    z_min = torch.clamp(
        oil_nodes_downsampled[:, 2] - half_patch,
        0,
        tif_data.shape[2] - patch_size,  # noqa E501
    )

    # Create a grid of offsets for an 8x8x8 patch.
    offsets = torch.arange(patch_size, device=tif_data.device)
    # print('offsets', offsets)
    x_grid, y_grid, z_grid = torch.meshgrid(
        offsets, offsets, offsets, indexing="ij"
    )  # noqa E501
    # print('x_grid', x_grid)
    # print('z_min', z_min)

    # Broadcast the starting indices over the patch grid dimensions
    x_idx = x_min[:, None, None, None] + x_grid  # (N, 8, 8, 8)
    y_idx = y_min[:, None, None, None] + y_grid
    z_idx = z_min[:, None, None, None] + z_grid

    # Use advanced indexing to extract all patches in one operation.
    patches = tif_data[x_idx, y_idx, z_idx]

    return patches.float()[:, ::ds, ::ds, ::ds]


def get_face_angles(points, faces, return_obtuse_marks=False):
    """
    Calculate the rad of the angles of each
    triangle in the mesh.
    """
    triangles = points[faces]
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]

    AB = B - A
    AC = C - A
    BC = C - B
    BA = A - B
    CA = A - C
    CB = B - C

    # Calculate lengths with epsilon for numerical stability
    epsilon = 1e-10
    AB_len = np.linalg.norm(AB, axis=1) + epsilon
    AC_len = np.linalg.norm(AC, axis=1) + epsilon
    BC_len = np.linalg.norm(BC, axis=1) + epsilon

    # Calculate angles with epsilon for numerical stability
    theta_A = np.arccos(
        np.clip(
            np.sum(AB * AC, axis=1) / (AB_len * AC_len),
            -1 + epsilon,
            1 - epsilon,
        )  # noqa E501
    )
    theta_B = np.arccos(
        np.clip(
            np.sum(BA * BC, axis=1) / (AB_len * BC_len),
            -1 + epsilon,
            1 - epsilon,
        )  # noqa E501
    )
    theta_C = np.arccos(
        np.clip(
            np.sum(CA * CB, axis=1) / (AC_len * BC_len),
            -1 + epsilon,
            1 - epsilon,
        )  # noqa E501
    )

    obtuse_marks = np.zeros(len(faces), dtype=int)
    # Check each angle
    obtuse_mask_A = theta_A > np.pi / 2
    obtuse_mask_B = theta_B > np.pi / 2
    obtuse_mask_C = theta_C > np.pi / 2

    # Set obtuse vertex index
    obtuse_marks[obtuse_mask_A] = faces[obtuse_mask_A, 0]
    obtuse_marks[obtuse_mask_B] = faces[obtuse_mask_B, 1]
    obtuse_marks[obtuse_mask_C] = faces[obtuse_mask_C, 2]

    if return_obtuse_marks:
        return theta_A, theta_B, theta_C, obtuse_marks
    else:
        return theta_A, theta_B, theta_C


def interpolate_shape(V0, V1, t=0.5):
    sdf0 = distance_transform_edt(V0) - distance_transform_edt(~V0)
    sdf1 = distance_transform_edt(V1) - distance_transform_edt(~V1)
    sdf_t = (1 - t) * sdf0 + t * sdf1
    # V_t = (sdf_t >= 0).astype(np.uint8)
    V_t = (sdf_t >= -0.25).astype(np.uint8)
    return V_t


def interpolate_velocity_field(vel, pore_mask, boundary_ratio=3.0):
    """
    Interpolate a sparse velocity field while respecting rock boundaries,
    ensuring velocities at pore walls are exactly zero.

    Parameters:
        vel (np.ndarray): Sparse velocity field of shape (3, D, H, W)
        pore_mask (np.ndarray): Binary mask where 1 indicates fluid,
        0 indicates rock or other phase
        sigma (float): Sigma value for Gaussian smoothing
        boundary_ratio (float): Ratio of boundary points to non-zero velocity points.  # noqa
            Default 1.0 means equal number of boundary points as non-zero points.
            Lower values reduce computation time but may affect accuracy.
    """
    # Create output array
    vel_interp = np.zeros_like(vel)

    # Create grid once for all components
    grid_z, grid_y, grid_x = np.mgrid[
        0 : vel.shape[1],
        0 : vel.shape[2],
        0 : vel.shape[3],  # noqa E501
    ]
    grid_coords = (grid_z, grid_y, grid_x)

    # Find boundary points (pore voxels adjacent to rock)
    rock_mask = 1 - pore_mask
    dilated_rock = ndi.binary_dilation(rock_mask, iterations=1)
    boundary_mask = dilated_rock & pore_mask

    # Process all components in parallel where possible
    for i in range(3):
        # Get points and values for this component
        non_zero = vel[i] != 0
        if non_zero.sum() > 0:
            # Get particle measurement points
            points = np.array(np.where(non_zero)).T
            values = vel[i][non_zero]

            # Get boundary points
            boundary_points = np.array(np.where(boundary_mask)).T

            # Calculate number of boundary points to use based on ratio
            n_boundary_points = int(len(points) * boundary_ratio)

            if n_boundary_points < len(boundary_points):
                # Randomly sample boundary points if we have too many
                indices = np.random.choice(
                    len(boundary_points),
                    size=n_boundary_points,
                    replace=False,
                )
                boundary_points = boundary_points[indices]

            # Add boundary points with zero velocity
            boundary_values = np.zeros(len(boundary_points))

            # Combine particle points and boundary points
            combined_points = np.vstack((points, boundary_points))
            combined_values = np.hstack((values, boundary_values))

            # Interpolate with both real measurements and boundary constraints
            vel_interp[i] = griddata(
                combined_points,
                combined_values,
                grid_coords,
                method="linear",
                fill_value=0,
            )
            # vel_interp[i] = griddata(
            #     combined_points,
            #     combined_values,
            #     grid_coords,
            #     method="nearest",
            #     fill_value=0,
            # )

    # Apply pore mask (vectorized operation)
    vel_interp *= pore_mask[np.newaxis, :, :, :]

    return vel_interp


def calculate_velocity_magnitude(vel):
    """Calculate the magnitude of a velocity field vectorized."""
    return np.sqrt(np.sum(vel**2, axis=0))


def normalize_velocity_data(velocity, velocity_stats):
    """Normalize velocity data using pre-calculated statistics"""
    # Create mask for valid velocity measurements
    valid_mask = velocity != 0

    if valid_mask.sum() == 0:
        return velocity

    # Normalize valid measurements in-place
    valid_velocities = velocity[valid_mask]
    velocity[valid_mask] = (valid_velocities - velocity_stats["min"]) / (
        velocity_stats["max"] - velocity_stats["min"] + 1e-8
    )
    return velocity


def cube2mesh(
    cube,
    threshold=1,
    smooth=True,
    smooth_iter=10,
    relaxation_factor=0.1,
):  # noqa E501
    """
    Extract the geometry of the surface from a tif data.
    as well as surface smoothing.
    """
    cube = cube == threshold
    verts, faces, _, _ = measure.marching_cubes(cube, level=0)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(
        np.int64
    )  # noqa E501
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    if smooth:
        mesh = mesh.smooth(
            n_iter=smooth_iter, relaxation_factor=relaxation_factor
        )  # noqa E501
    mesh.face_raw = faces
    return mesh


def down_sample(cube, factor):
    if cube.ndim == 4:
        # 4D case: (C, D, H, W)
        C, D, H, W = cube.shape
        new_D = (D // factor) * factor
        new_H = (H // factor) * factor
        new_W = (W // factor) * factor
        sliced = cube[:, :new_D, :new_H, :new_W][
            :, ::factor, ::factor, ::factor
        ]  # noqa E501
        return sliced
    elif cube.ndim == 3:
        # 3D case: (D, H, W)
        D, H, W = cube.shape
        new_D = (D // factor) * factor
        new_H = (H // factor) * factor
        new_W = (W // factor) * factor
        sliced = cube[:new_D, :new_H, :new_W][::factor, ::factor, ::factor]
        return sliced
    else:
        raise ValueError("Input cube must be either 3D or 4D.")


# def remesh_surf(coords, faces, lc=0.1):
#     with pygmsh.geo.Geometry() as geom:
#         # 1) create all gmsh points
#         gmsh_pts = [
#             geom.add_point([x, y, z], lc)
#             for x, y, z in coords
#         ]
#         # 2) for each triangle, make a plane surface
#         #    (you’ll end up with one surface per face; Gmsh will stitch them)
#         for tri in faces:
#             # create the three line segments
#             l1 = geom.add_line(gmsh_pts[tri[0]], gmsh_pts[tri[1]])
#             l2 = geom.add_line(gmsh_pts[tri[1]], gmsh_pts[tri[2]])
#             l3 = geom.add_line(gmsh_pts[tri[2]], gmsh_pts[tri[0]])
#             # loop + surface
#             loop = geom.add_curve_loop([l1, l2, l3])
#             geom.add_plane_surface(loop)
#         # 3) generate a new mesh
#         mesh = geom.generate_mesh(dim=2, algorithm=1)
#     points = mesh.points
#     faces = mesh.cells_dict["triangle"]
#     return points, faces


def geo2pvmesh(coords, faces):
    faces_pv = np.hstack(
        [
            np.full((faces.shape[0], 1), 3, dtype=np.int64),
            faces.astype(np.int64),
        ]  # noqa E501
    ).flatten()
    mesh_pv = pv.PolyData(coords, faces_pv)
    mesh_pv.face_raw = faces
    return mesh_pv


def pvmesh2geo(mesh_pv):
    """Convert PyVista mesh back to coordinates and faces format.
    Parameters:
    -----------
    mesh_pv : pv.PolyData
        PyVista mesh object

    Returns:
    --------
    coords : ndarray
        Vertex coordinates
    faces : ndarray
        Face connectivity array
    """
    coords = mesh_pv.points
    # Parse PyVista faces format: [n, v1, v2, v3, n, v4, v5, v6, ...]
    pv_faces = mesh_pv.faces
    faces = pv_faces.reshape(-1, 4)[:, 1:]  # Remove count, keep vertices
    return coords, faces


# The numpy max pooling function optimized with Numba
@numba.njit(parallel=True)
def max_pool3d_numba(arr, factor):
    """
    Numba-optimized version of max_pool3d that is much faster than the numpy
    version.

    Parameters:
        arr (numpy.ndarray): Input array of shape (C, D, H, W)
        factor (int): Downsampling factor along each spatial axis.

    Returns:
        numpy.ndarray: Downsampled array with shape (C, D//factor, H//factor,
        W//factor)
    """
    C, D, H, W = arr.shape
    new_D = D // factor
    new_H = H // factor
    new_W = W // factor

    # Create output array
    pooled = np.zeros((C, new_D, new_H, new_W), dtype=arr.dtype)

    # Parallel processing over each channel
    for c in numba.prange(C):
        for d in range(new_D):
            for h in range(new_H):
                for w in range(new_W):
                    # Get the current block
                    d_start = d * factor
                    h_start = h * factor
                    w_start = w * factor

                    # Find max and min in the block
                    block_max = arr[c, d_start, h_start, w_start]
                    block_min = block_max
                    max_abs = abs(block_max)
                    for dd in range(factor):
                        for hh in range(factor):
                            for ww in range(factor):
                                val = arr[
                                    c,
                                    d_start + dd,
                                    h_start + hh,
                                    w_start + ww,
                                ]  # noqa E501
                                abs_val = abs(val)

                                if abs_val > max_abs:
                                    max_abs = abs_val
                                    block_max = val

                                if val < block_min:
                                    block_min = val

                    # Select the value with the larger absolute value
                    if abs(block_max) >= abs(block_min):
                        pooled[c, d, h, w] = block_max
                    else:
                        pooled[c, d, h, w] = block_min

    return pooled


def _max_pool3d_numpy_vectorized(arr_cropped, C, new_D, new_H, new_W, factor):
    """
    Optimized vectorized max pooling for moderate-sized arrays.
    Uses more efficient memory access patterns and operations.
    """
    # Use contiguous memory layout for better cache performance
    arr_reshaped = np.ascontiguousarray(
        arr_cropped.reshape(C, new_D, factor, new_H, factor, new_W, factor)
    )

    # Compute max/min more efficiently
    pooled_max = arr_reshaped.max(axis=(2, 4, 6))
    pooled_min = arr_reshaped.min(axis=(2, 4, 6))

    # Optimize the comparison using direct boolean indexing
    abs_max = np.abs(pooled_max)
    abs_min = np.abs(pooled_min)

    # Direct boolean indexing is faster than np.where for this use case
    pooled = pooled_max.copy()
    mask = abs_max < abs_min
    pooled[mask] = pooled_min[mask]

    return pooled


def _max_pool3d_numpy_chunked(arr_cropped, C, new_D, new_H, new_W, factor):
    """
    Memory-efficient chunked processing for very large arrays.
    Processes the array in smaller chunks to avoid memory issues.
    """
    pooled = np.zeros((C, new_D, new_H, new_W), dtype=arr_cropped.dtype)

    # Process in chunks along the depth dimension
    chunk_size = min(50, new_D)  # Adaptive chunk size

    for d_start in range(0, new_D, chunk_size):
        d_end = min(d_start + chunk_size, new_D)
        d_slice_start = d_start * factor
        d_slice_end = d_end * factor

        # Extract and reshape chunk
        chunk = arr_cropped[:, d_slice_start:d_slice_end, :, :]
        chunk_reshaped = chunk.reshape(
            C, d_end - d_start, factor, new_H, factor, new_W, factor
        )

        # Compute max/min for this chunk
        chunk_max = chunk_reshaped.max(axis=(2, 4, 6))
        chunk_min = chunk_reshaped.min(axis=(2, 4, 6))

        # Efficient comparison
        abs_max = np.abs(chunk_max)
        abs_min = np.abs(chunk_min)

        # Direct assignment
        chunk_pooled = chunk_max
        mask = abs_max < abs_min
        chunk_pooled[mask] = chunk_min[mask]

        pooled[:, d_start:d_end, :, :] = chunk_pooled

    return pooled


# Try to import numba for optional acceleration
try:
    import numba

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _max_pool3d_numpy_numba(arr_cropped, C, new_D, new_H, new_W, factor):
        """
        Numba-accelerated version for maximum performance.
        This is 10-100x faster than pure numpy for large arrays.
        """
        pooled = np.zeros((C, new_D, new_H, new_W), dtype=arr_cropped.dtype)
        for d in numba.prange(new_D):
            for c in range(C):
                # for c in numba.prange(C):  # Parallel over channels
                # for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        d_start = d * factor
                        h_start = h * factor
                        w_start = w * factor

                        # Initialize with first element
                        max_val = arr_cropped[c, d_start, h_start, w_start]
                        min_val = max_val

                        # Find element with largest absolute value
                        for dd in range(factor):
                            for hh in range(factor):
                                for ww in range(factor):
                                    idx_d = d_start + dd
                                    idx_h = h_start + hh
                                    idx_w = w_start + ww
                                    val = arr_cropped[c, idx_d, idx_h, idx_w]
                                    if val > max_val:
                                        max_val = val
                                    if val < min_val:
                                        min_val = val

                        # Select based on absolute value
                        if abs(max_val) >= abs(min_val):
                            pooled[c, d, h, w] = max_val
                        else:
                            pooled[c, d, h, w] = min_val

        return pooled

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    _max_pool3d_numpy_numba = None


def max_pool3d_numpy(arr, factor, backend="numpy"):
    """
    Perform pooling (downsampling) in 3D using NumPy or PyTorch.

    If backend is 'torch', uses torch.nn.functional.max_pool3d for
    acceleration.

    For each block, selects the element (max or min) with the larger
    absolute value.

    Parameters:
        arr (numpy.ndarray or torch.Tensor): Input array of shape (C, D, H, W)
        factor (int): Downsampling factor along each spatial axis.
        backend (str): 'numpy' or 'torch'

    Returns:
        numpy.ndarray or torch.Tensor: Downsampled array with shape
            (C, D//factor, H//factor, W//factor)
    """
    input_is_numpy = isinstance(arr, np.ndarray)

    if backend == "torch":
        if input_is_numpy:
            arr = torch.from_numpy(arr)

        C, D, H, W = arr.shape
        total_elements = C * D * H * W

        # Determine if we should use half precision
        use_fp16 = total_elements > 5e8  # Use fp16 for arrays > 500M elements

        # Smart device selection based on array size
        if torch.cuda.is_available():
            # Adjust memory estimate based on precision
            bytes_per_element = 2 if use_fp16 else 4  # fp16 = 2 bytes, fp32 = 4 bytes
            bytes_needed = (
                total_elements * bytes_per_element * 2.5
            )  # 2.5x for temp arrays
            gpu_mem = torch.cuda.get_device_properties(0).total_memory

            if bytes_needed < gpu_mem * 0.7:  # Use 70% of GPU memory max
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                print(
                    f"Array too large for GPU ({bytes_needed/1e9:.1f}GB needed), using CPU"
                )
        else:
            device = torch.device("cpu")

        new_D = D // factor
        new_H = H // factor
        new_W = W // factor

        arr_cropped = arr[:, : new_D * factor, : new_H * factor, : new_W * factor]

        # Convert to half precision if needed
        original_dtype = arr_cropped.dtype
        if use_fp16:
            arr_cropped = arr_cropped.half()  # Convert to float16
            print(
                f"Using fp16 to save memory (was {total_elements*4/1e9:.1f}GB, now {total_elements*2/1e9:.1f}GB)"
            )

        # Memory optimization strategies based on size
        if total_elements > 2e9:  # > 2 billion elements
            # Strategy 1: Process channel by channel with fp16
            pooled_list = []
            for c in range(C):
                # Process one channel at a time
                ch_data = (
                    arr_cropped[c : c + 1].to(device).unsqueeze(0)
                )  # (1, 1, D, H, W)

                # Use mixed precision if available
                if device.type == "cuda" and use_fp16:
                    with torch.cuda.amp.autocast():
                        ch_max = F.max_pool3d(
                            ch_data, kernel_size=factor, stride=factor
                        )
                        ch_data.neg_()
                        ch_min = -F.max_pool3d(
                            ch_data, kernel_size=factor, stride=factor
                        )
                else:
                    ch_max = F.max_pool3d(ch_data, kernel_size=factor, stride=factor)
                    ch_data.neg_()
                    ch_min = -F.max_pool3d(ch_data, kernel_size=factor, stride=factor)

                # Select based on absolute value
                ch_pooled = torch.where(
                    torch.abs(ch_max) >= torch.abs(ch_min), ch_max, ch_min
                ).squeeze(0)

                # Convert back to original precision before storing
                if use_fp16 and original_dtype == torch.float32:
                    ch_pooled = ch_pooled.float()

                pooled_list.append(ch_pooled.cpu())

                # Explicitly free GPU memory
                del ch_data, ch_max, ch_min, ch_pooled
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            pooled = torch.cat(pooled_list, dim=0)

        elif total_elements > 5e8:  # > 500M elements
            # Strategy 2: Use in-place operations with fp16
            arr_cropped = arr_cropped.to(device)
            arr_cropped = arr_cropped.unsqueeze(0)  # (1, C, D, H, W)

            if device.type == "cuda" and use_fp16:
                with torch.cuda.amp.autocast():
                    pooled_max = F.max_pool3d(
                        arr_cropped, kernel_size=factor, stride=factor
                    )
                    arr_cropped.neg_()
                    pooled_min = -F.max_pool3d(
                        arr_cropped, kernel_size=factor, stride=factor
                    )
            else:
                pooled_max = F.max_pool3d(
                    arr_cropped, kernel_size=factor, stride=factor
                )
                arr_cropped.neg_()
                pooled_min = -F.max_pool3d(
                    arr_cropped, kernel_size=factor, stride=factor
                )

            pooled_max = pooled_max.squeeze(0)
            pooled_min = pooled_min.squeeze(0)

            pooled = torch.where(
                torch.abs(pooled_max) >= torch.abs(pooled_min),
                pooled_max,
                pooled_min,
            )

            # Convert back to original precision if needed
            if use_fp16 and original_dtype == torch.float32:
                pooled = pooled.float()

            if device.type == "cuda":
                pooled = pooled.cpu()

        else:
            # Strategy 3: Standard approach for smaller arrays (keep full precision)
            arr_cropped = arr_cropped.to(device)
            arr_cropped = arr_cropped.unsqueeze(0)  # (1, C, D, H, W)

            pooled_max = F.max_pool3d(arr_cropped, kernel_size=factor, stride=factor)
            pooled_min = -F.max_pool3d(-arr_cropped, kernel_size=factor, stride=factor)

            pooled_max = pooled_max.squeeze(0)
            pooled_min = pooled_min.squeeze(0)

            pooled = torch.where(
                torch.abs(pooled_max) >= torch.abs(pooled_min),
                pooled_max,
                pooled_min,
            )

            if device.type == "cuda":
                pooled = pooled.cpu()

        if input_is_numpy:
            return pooled.numpy()
        else:
            return pooled

    elif backend == "numpy":
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        C, D, H, W = arr.shape
        new_D = D // factor
        new_H = H // factor
        new_W = W // factor

        arr_cropped = arr[:, : new_D * factor, : new_H * factor, : new_W * factor]

        # Optimized approach: Use best available method
        # Priority: numba (if available) > chunked (for large) > vectorized
        total_elements = C * D * H * W
        use_chunked = total_elements > 100_000_000  # 100M elements threshold

        if NUMBA_AVAILABLE and _max_pool3d_numpy_numba is not None:
            # Use numba for maximum performance (10-100x faster)
            # pooled = _max_pool3d_numpy_vectorized(
            #     arr_cropped, C, new_D, new_H, new_W, factor
            # )
            pooled = _max_pool3d_numpy_numba(
                arr_cropped, C, new_D, new_H, new_W, factor
            )
        elif use_chunked:
            # Chunked processing for very large arrays without numba
            pooled = _max_pool3d_numpy_chunked(
                arr_cropped, C, new_D, new_H, new_W, factor
            )
        else:
            # Optimized vectorized approach for smaller arrays
            pooled = _max_pool3d_numpy_vectorized(
                arr_cropped, C, new_D, new_H, new_W, factor
            )

        return pooled

    else:
        raise ValueError(f"Backend {backend} not supported")


# Keep the original numpy version for reference
# def max_pool3d_numpy(arr, factor, backend="numpy"):
#     """
#     Perform pooling (downsampling) in 3D using NumPy, selecting, for each block,  # noqa E501
#     the element (either maximum or minimum) with the largest absolute value.

#     Parameters:
#         arr (numpy.ndarray): Input array of shape (C, D, H, W)
#         factor (int): Downsampling factor along each spatial axis.

#     Returns:
#         numpy.ndarray: Downsampled array with shape (C, D//factor, H//factor, W//factor)  # noqa E501
#                       where each element is chosen between the max and min of the block  # noqa E501
#                       based on which has the larger absolute value.
#     """
#     if backend == "numpy":
#         C, D, H, W = arr.shape
#         new_D = D // factor
#         new_H = H // factor
#         new_W = W // factor

#         # Crop the array so that dimensions are exactly divisible by factor.
#         arr_cropped = arr[:, : new_D * factor, : new_H * factor, : new_W * factor]

#         # Reshape to create pooling blocks.
#         # New shape: (C, new_D, factor, new_H, factor, new_W, factor)
#         arr_reshaped = arr_cropped.reshape(
#             C, new_D, factor, new_H, factor, new_W, factor
#         )  # noqa E501

#         # Compute maximum and minimum in each block along the block dimensions.
#         pooled_max = arr_reshaped.max(axis=(2, 4, 6))
#         pooled_min = arr_reshaped.min(axis=(2, 4, 6))

#         # Compare the absolute values and select the corresponding value.
#         pooled = np.where(
#             np.abs(pooled_max) >= np.abs(pooled_min),
#             pooled_max,
#             pooled_min,
#         )  # noqa E501
#         return pooled
#     elif backend == "torch":
#         pass
#     else:
#         raise ValueError(f"Backend {backend} not supported")


def process_frame(
    df,
    frame,
    grid_size,
    x_key,
    y_key,
    z_key,
    frame_key,
    vx_key,
    vy_key,
    vz_key,
    offset=[50, 50, 0],
):
    """
    Compute the physics grid for a single frame.

    The grid has 3 channels (for vx, vy, vz) and a spatial shape of grid_size.
    It uses the logic from your dataset method:
      - x_idx and y_idx are computed as (value + 50)
      - z_idx is computed as (value + 50) and then subtracted by 100.

    Parameters:
        df (pd.DataFrame): Dataframe containing the velocity data.
        frame: Value identifying the current frame.
        grid_size (tuple): Spatial grid size (D, H, W).
        x_key, y_key, z_key, frame_key (str): Column names for coordinates and
        frame. vx_key, vy_key, vz_key (str): Column names for
        the velocity components.

    Returns:
        numpy.ndarray: Grid of shape (3, D, H, W) with vx, vy, vz placed at
        the indexed locations.
    """
    off_x, off_y, off_z = offset
    grid = np.zeros((3,) + grid_size, dtype=np.float32)
    dff = df[df[frame_key] == frame].copy()

    dff[f"{x_key}_idx"] = (dff[x_key]).astype(int)
    dff[f"{y_key}_idx"] = (dff[y_key]).astype(int)
    dff[f"{z_key}_idx"] = (dff[z_key]).astype(int)

    z_idx = dff[f"{z_key}_idx"].values + off_z
    y_idx = dff[f"{y_key}_idx"].values + off_y
    x_idx = dff[f"{x_key}_idx"].values + off_x

    grid[0, z_idx, y_idx, x_idx] = dff[vx_key].values
    grid[1, z_idx, y_idx, x_idx] = dff[vy_key].values
    grid[2, z_idx, y_idx, x_idx] = dff[vz_key].values

    return grid


def gns_out_to_grid(
    velocity,
    positions,
    grid_size,
    offset=[50, 50, 0],
):
    """
    For GNS output, the channel order is z, y, x.
    Compute the physics grid for a single frame.

    Returns:
        numpy.ndarray: Grid of shape (3, D, H, W) with vx, vy, vz placed at
        the indexed locations.
    """
    # print("velocity shape:", velocity.shape)  # shape [12882, 3]
    # print("positions shape:", positions.shape)  # shape [12882, 3]
    # print("grid_size shape:", grid_size)
    offset_x, offset_y, offset_z = offset
    grid = np.zeros((3,) + grid_size, dtype=np.float32)
    positions = positions.cpu().numpy().astype(int)
    velocity = velocity.cpu().numpy()
    positions[:, 0] += offset_z
    positions[:, 1] += offset_y
    positions[:, 2] += offset_x
    positions = positions.astype(int)

    z_idx = positions[:, 0]
    y_idx = positions[:, 1]
    x_idx = positions[:, 2]

    # Clamp those particles that are out of the grid
    z_idx = np.clip(positions[:, 0] + offset[2], 0, grid_size[0] - 1)
    y_idx = np.clip(positions[:, 1] + offset[1], 0, grid_size[1] - 1)
    x_idx = np.clip(positions[:, 2] + offset[0], 0, grid_size[2] - 1)

    grid[0, z_idx, y_idx, x_idx] = velocity[:, 2]
    grid[1, z_idx, y_idx, x_idx] = velocity[:, 1]
    grid[2, z_idx, y_idx, x_idx] = velocity[:, 0]

    print("vel interpolated grid shape:", grid.shape)

    return grid
