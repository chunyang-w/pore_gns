"""
Minkowski functionals M0, M1, M2, M3
M0: volume
M1: surface area
M2: mean curvature
M3: Gaussian curvature
"""

import numpy as np
import time

from pore_net.utils import get_face_angles


def cotangent(points, a, b, c, epsilon=1e-12):
    # Calculate edge vectors
    ba = points[b] - points[a]
    ca = points[c] - points[a]

    # Dot product
    dot = np.dot(ba, ca)

    # Efficient 2-norm for cross product (no sqrt)
    cross = np.cross(ba, ca)
    cross_norm_sq = np.dot(cross, cross)  # ||u x v||^2

    # Use soft clipping to avoid zero division
    res = dot / np.sqrt(cross_norm_sq + epsilon)
    return res


def vectorized_cotangent(
    points, a_indices, b_indices, c_indices, epsilon=1e-12
):  # noqa E501
    """
    Vectorized cotangent calculation for multiple angle sets at once.

    Parameters:
    -----------
    points : array
        Mesh vertices
    a_indices : array
        Array of vertex indices for point a
    b_indices : array
        Array of vertex indices for point b
    c_indices : array
        Array of vertex indices for point c
    epsilon : float
        Small value to prevent division by zero

    Returns:
    --------
    array
        Cotangent values for all input points
    """
    # Get the actual points from indices
    a = points[a_indices]
    b = points[b_indices]
    c = points[c_indices]

    # Calculate edge vectors for all points at once
    ba = b - a
    ca = c - a

    # Calculate dot products for all sets at once
    # Sum along axis 1 to get scalar dot product for each vertex pair
    dots = np.sum(ba * ca, axis=1)

    # Calculate cross products for all sets at once
    crosses = np.cross(ba, ca)

    # Calculate squared norms of cross products
    cross_norms_sq = np.sum(crosses * crosses, axis=1)

    # Calculate and return cotangents
    return dots / np.sqrt(cross_norms_sq + epsilon)


def M_0(binary_image, scale_unit=1):
    """Compute the 0th Minkowski functional (volume).

    Parameters:
    -----------
    binary_image : ndarray
        3D binary image

    Returns:
    --------
    float
        The volume (number of nonzero voxels)
    """
    return np.sum(binary_image) * scale_unit**3


# Mesh-based Minkowski functionals
def M_1_mesh(mesh, scale_unit=1):
    """Compute the 1st Minkowski functional (surface area).

    M1(X) = ∫ δX ds
    Integral of surface elements over the boundary surface.

    Parameters:
    -----------
    mesh : object
        Mesh with points and face_raw attributes

    Returns:
    --------
    float
        Total surface area of the mesh
    """
    points = mesh.points
    points *= scale_unit
    faces = mesh.face_raw

    # Calculate area of each triangular face
    total_area = 0.0
    for face in faces:
        # Get vertices of the triangle
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]

        # Calculate two edges of the triangle
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Area = 0.5 * |cross product of edges|
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        total_area += area

    return total_area


def M_2_diff(mesh, unit_scale=1, verbose=False, use_filter=False):
    """
    Compute the 2nd Minkowski functional (mean curvature) using
    differential geometry.

    M2(X) = ∫ δX [1/r1 + 1/r2]ds
    Integral of mean curvature over boundary surface.

    Parameters:
    -----------
    mesh : object
        Mesh with points and face_raw attributes
    unit_scale : float, optional
        Scale factor for the mesh points, by default 1
    verbose : bool, optional
        Whether to print progress and timing information, by default False

    Returns:
    --------
    float
        Integrated mean curvature
    """
    points = mesh.points
    points *= unit_scale
    faces = mesh.face_raw

    # Build vertex to faces mapping
    start_time = time.time()
    vertex_to_faces = [[] for _ in range(len(points))]
    for i, face in enumerate(faces):
        for v in face:
            vertex_to_faces[v].append(i)
    if verbose:
        print(
            f"Built vertex to faces mapping ({time.time() - start_time:.2f}s)"
        )  # noqa E501

    # Calculate face normals and areas
    start_time = time.time()
    face_normals = np.zeros((len(faces), 3))
    face_areas = np.zeros(len(faces))

    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    # Calculate edges for all faces
    edge1 = v1 - v0
    edge2 = v2 - v0
    # Calculate cross products for all faces
    cross = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    face_normals = np.zeros_like(cross)
    mask = face_areas > 0

    face_normals[mask] = cross[mask] / (2 * face_areas[mask, np.newaxis])
    if verbose:
        print(
            f"Calculated face normals and areas ({time.time() - start_time:.2f}s)"
        )  # noqa E501

    # Calculate vertex normals using Eq. (3)
    start_time = time.time()
    vertex_normals = np.zeros((len(points), 3))
    for i in range(len(points)):
        normal_sum = np.zeros(3)
        for face_idx in vertex_to_faces[i]:
            normal_sum += face_normals[face_idx]

        norm = np.linalg.norm(normal_sum)
        if norm > 0:
            vertex_normals[i] = normal_sum / norm
    if verbose:
        print(f"Calculated vertex normals ({time.time() - start_time:.2f}s)")

    # Create edge to vertices mapping for cotangent calculation
    start_time = time.time()
    edges = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            v2 = face[(i + 2) % 3]

            # Add edge and opposite vertex
            edge = tuple(sorted([v0, v1]))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append((face_idx, v2))
    if verbose:
        print(
            f"Created edge to vertices mapping ({time.time() - start_time:.2f}s)"
        )  # noqa E501

    # Calculate angles
    start_time = time.time()
    angle_A, angle_B, angle_C, obtuse_marks = get_face_angles(
        points, faces, return_obtuse_marks=True
    )

    if verbose:
        print(f"Calculated face angles ({time.time() - start_time:.2f}s)")

    # calculate voronoi areas
    voronoi_areas = np.zeros(len(points))
    for i in range(len(points)):
        for face_idx in vertex_to_faces[i]:
            face_area = face_areas[face_idx]
            if obtuse_marks[face_idx] == 0:
                voronoi_areas[i] += face_area / 3
            elif obtuse_marks[face_idx] == i:  # angle at vertex is obtuse
                voronoi_areas[i] += face_area / 2
            else:  # angle at vertex is acute
                voronoi_areas[i] += face_area / 4

    # Calculate mean curvatures using Meyer et al. method
    start_time = time.time()
    K_operators = np.zeros((len(points), 3))

    # Convert dictionary to arrays for vectorized processing
    valid_edges = []
    opposite_pairs = []
    edge_endpoints = []

    start_time = time.time()
    for edge, opposite_verts in edges.items():
        if len(opposite_verts) == 2:
            valid_edges.append(edge)
            opposite_pairs.append((opposite_verts[0][1], opposite_verts[1][1]))
            edge_endpoints.append(edge)
    if verbose:
        print(
            f"Created valid edges and pairs ({time.time() - start_time:.2f}s)"
        )  # noqa E501

    valid_edges = np.array(valid_edges)
    v0_indices = np.array([e[0] for e in edge_endpoints])
    v1_indices = np.array([e[1] for e in edge_endpoints])
    opposite1_indices = np.array([op[0] for op in opposite_pairs])
    opposite2_indices = np.array([op[1] for op in opposite_pairs])

    # Calculate cotangents all at once
    cot_alphas = vectorized_cotangent(points, opposite1_indices, v0_indices, v1_indices)
    cot_betas = vectorized_cotangent(points, opposite2_indices, v0_indices, v1_indices)

    # Calculate edge vectors and lengths
    edge_vecs = points[v0_indices] - points[v1_indices]
    # edge_len_squared = np.sum(edge_vecs**2, axis=1)

    # Calculate weights
    # weights = 0.125 * (cot_alphas + cot_betas) * edge_len_squared
    cot_weights = cot_alphas + cot_betas

    # Update voronoi areas and K operators
    for i in range(len(valid_edges)):
        v0, v1 = v0_indices[i], v1_indices[i]
        # voronoi_areas[v0] += weights[i]
        # voronoi_areas[v1] += weights[i]
        K_operators[v0] += cot_weights[i] * edge_vecs[i]
        K_operators[v1] -= cot_weights[i] * edge_vecs[i]
    if verbose:
        print(
            f"Calculated Voronoi areas and K operators ({time.time() - start_time:.2f}s)"
        )  # noqa E501

    # Calculate mean curvatures using Eq. (6)
    start_time = time.time()
    mean_curvatures = np.zeros(len(points))
    for i in range(len(points)):
        if voronoi_areas[i] > 0:
            # Normalize by Voronoi area
            K = K_operators[i] / (2 * voronoi_areas[i])
            # Calculate mean curvature with dot product (Eq. 6)
            mean_curvatures[i] = 0.5 * np.dot(K, vertex_normals[i])
    if verbose:
        print(f"Calculated mean curvatures ({time.time() - start_time:.2f}s)")

    # Calculate total mesh area
    total_area = np.sum(face_areas)  # noqa

    # Calculate M2 (integrated mean curvature) using Eq. (7)\
    if use_filter:
        # Calculate mean values
        mean_curvatures_mean = np.mean(mean_curvatures)
        # voronoi_areas_mean = np.mean(voronoi_areas)

        # Calculate 5th and 95th percentiles
        mean_curv_lower = np.percentile(mean_curvatures, 1)
        mean_curv_higher = np.percentile(mean_curvatures, 99)
        # voronoi_lower = np.percentile(voronoi_areas, 1)
        # voronoi_higher = np.percentile(voronoi_areas, 99)

        # Filter outliers: replace values outside percentile range
        outlier_mask = (mean_curvatures < mean_curv_lower) | (
            mean_curvatures > mean_curv_higher
        )
        mean_curvatures[outlier_mask] = mean_curvatures_mean

        # outlier_mask = (voronoi_areas < voronoi_lower) | \
        #                (voronoi_areas > voronoi_higher)
        # voronoi_areas[outlier_mask] = voronoi_areas_mean

    M2 = np.sum(mean_curvatures * voronoi_areas)
    # M2 = np.sum(mean_curvatures)
    avg_mean_curvature = M2 / total_area
    return M2, mean_curvatures, avg_mean_curvature


def M_3_mesh(mesh):
    """Compute the 3rd Minkowski functional (Gaussian curvature).

    M3(X) = ∫ δX [1/(r1*r2)]ds = 2πχ(δX) = 4πχ(X)
    Where χ is the Euler characteristic.

    Parameters:
    -----------
    mesh : object
        Mesh with points and face_raw attributes

    Returns:
    --------
    float
        Integrated Gaussian curvature (2πχ)
    """
    points = mesh.points
    faces = mesh.face_raw

    # Get number of vertices, edges, and faces
    V = len(points)
    F = len(faces)

    # Build edge set
    edges = set()
    for face in faces:
        # Add all three edges of the triangle
        # Sort vertices to ensure consistent edge representation
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    # Count edges
    E = len(edges)

    # Calculate Euler characteristic: V - E + F
    euler_characteristic = V - E + F

    # By Gauss-Bonnet theorem: ∫ K dA = 2πχ
    integrated_gaussian_curvature = 2 * np.pi * euler_characteristic

    return integrated_gaussian_curvature
