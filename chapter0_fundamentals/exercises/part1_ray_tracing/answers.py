#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
import torch as t
from torch import Tensor
import einops as e
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


# In[11]:


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    a = t.zeros(num_pixels, 1, 3)
    y_values = t.FloatTensor([-y_limit+(x/(num_pixels-1))*2*y_limit for x in range(num_pixels)])
    b = t.cat((t.ones(num_pixels, 1, 1), e.repeat(y_values, "a -> a 1 1"), t.zeros(num_pixels, 1, 1)), -1)
    return t.cat((a, b), 1)
    

# In[12]:
@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

# In[14]:


# @jaxtyped(typeguard.typechecked)
def intersect_ray_1d(ray: Float[Tensor, "2 3"], segment: Float[Tensor, "2 3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray = ray[:, 0:2]
    segment = segment[:, 0:2]
    direction = segment[0]-segment[1]
    stacked = t.stack((ray[1], direction), 1)
    try:
        result = t.linalg.solve(stacked, segment[0]-ray[0])
    except RuntimeError:
        return False
    return (result[0] >= 0 and result[1] <= 1 and result[1] >= 0).item()

# In[15]:

# In[16]:


def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''

    nrays = rays.shape[0]
    nsegments = segments.shape[0]

    rays2 = e.repeat(rays, "a b c -> a nsegments b c", nsegments=nsegments)
    segments2 = e.repeat(segments, "a b c -> nrays a b c", nrays=nrays)


    O = rays2[:, :, 0, :2]
    D = rays2[:, :, 1, :2]

    L1 = segments2[:, :, 0, :2]
    L2 = segments2[:, :, 1, :2]


    leftarray = t.stack((D, L1-L2), -1)
    
    rightarray = L1-O

    identity = e.repeat(t.eye(2,2), "a b -> nrays nsegments a b", nrays=nrays, nsegments=nsegments)

    mask = t.linalg.det(leftarray).abs() < 1e-6

    mask = e.repeat(mask, "a b -> a b 2 2")

    leftarray = t.where(mask, identity, leftarray)

    result = t.linalg.solve(leftarray, rightarray)
    
    result = (result[:, :, 0] >= 0) & (result[:, :, 1] <= 1) & (result[:, :, 1] >= 0)

    result = t.where(mask[:, :, 0, 0], False, result)

    result = e.reduce(result.long(), "a b -> a", "sum") > 0.5

    return result

# In[17]:


def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """


    a = t.zeros(num_pixels_y * num_pixels_z, 1, 3)
    y_vals = e.repeat(t.linspace(-y_limit, y_limit, num_pixels_y), "h -> (i h)", i=num_pixels_z)
    z_vals = e.repeat(t.linspace(-z_limit, z_limit, num_pixels_z), "h -> (h i)", i=num_pixels_y)
    b = t.stack([t.ones(num_pixels_y * num_pixels_z), y_vals, z_vals], -1)
    b = e.repeat(b, "a b -> a 1 b")
    return t.cat([a, b], 1)
    

    


    raise NotImplementedError()


# In[32]:


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)


# In[19]:


Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    M_l = t.stack([(-D), B-A, C-A], 1)
    M_r = O-A
    sol = t.linalg.solve(M_l, M_r)
    s, u, v = sol

    return (v.item() >= 0) and (u.item() >= 0) and (u.item()+v.item() <= 1)

# In[20]:


def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """

    nrays = rays.shape[0]
    triangles = e.repeat(triangle, "tp dim -> nrays tp dim", nrays=nrays)

    O = rays[:, 0, :]
    D = rays[:, 1, :]

    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    M_l = t.stack([(-D), B-A, C-A], 2)
    M_r = O-A

    mask = t.linalg.det(M_l).abs() < 1e-6

    M_l[mask] = t.eye(3)

    sol = t.linalg.solve(M_l, M_r)

    s, u, v = sol.unbind(1)
    
    return (v >= 0) & (u >= 0) & (u+v<= 1)  & (~mask)

    raise NotImplementedError()

# In[21]:


def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.shape[0]

    A, B, C = e.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], 2)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)

# In[22]:


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    

    NR = rays.shape[0]

    A, B, C = e.repeat(triangles, "ntr pts dims -> pts nrays ntr dims", nrays=NR)
    O, D = e.repeat(rays, "nrays raypoint dims -> raypoint nrays ntr dims", ntr=A.shape[1])

    mat = t.stack([- D, B - A, C - A], -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    s[~((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)] = float("inf")

    return e.reduce(s, "a b -> a", "min")

# In[23]:


def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """

    i = t.eye(3)
    i[[True, False, True], [True, False, True]] = t.cos(theta)
    i[2, 0] = -t.sin(theta)
    i[0, 2] = t.sin(theta)

    return i

# In[25]:


from tqdm import tqdm

def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix,
    raytrace_function,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z] element is distance
    to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


# In[48]:


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    
    rays = rays.cuda()
    triangles = triangles.cuda()

    NR = rays.shape[0]

    A, B, C = e.repeat(triangles, "ntr pts dims -> pts nrays ntr dims", nrays=NR)
    O, D = e.repeat(rays, "nrays raypoint dims -> raypoint nrays ntr dims", ntr=A.shape[1])

    mat = t.stack([- D, B - A, C - A], -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3).cuda()

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    s[~((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)] = float("inf")

    return e.reduce(s, "a b -> a", "min").cpu()

# In[57]:


def raytrace_wireframe(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    
    rays = rays.cuda()
    triangles = triangles.cuda()

    NR = rays.shape[0]

    A, B, C = e.repeat(triangles, "ntr pts dims -> pts nrays ntr dims", nrays=NR)
    O, D = e.repeat(rays, "nrays raypoint dims -> raypoint nrays ntr dims", ntr=A.shape[1])

    mat = t.stack([- D, B - A, C - A], -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3).cuda()

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    s[((((u == 0) | (v == 0)) & (u + v <= 1)) | ((u + v == 1) & (u >= 0) & (v >= 0)) & ~is_singular)] = float("inf")

    return e.reduce(s, "a b -> a", "max").cpu()


# In[ ]:




