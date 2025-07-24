from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from util.func import load_obj, make_sphere, make_star_cameras, normalize_vertices, save_obj
from util.render import NormalsRenderer
from tqdm import tqdm
from util.snapshot import snapshot
try:
    from util.view import show
except:
    show = None

from pathlib import Path
import imageio
import torch
import time

def save_images_new(images, dir):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()

    for i in range(images.shape[0]):
        img = (images[i] * 255).clamp(max=255).type(torch.uint8).numpy()

        # Handle grayscale (1 channel) by squeezing
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        imageio.imwrite(dir / f'{i:02d}.png', img)


fname = 'data/lucy.obj'
steps = 1000
snapshot_step = 1

# Setup cameras and renderer
mv, proj = make_star_cameras(4, 2)
renderer = NormalsRenderer(mv, proj, [512, 512])

# Load and normalize target mesh
target_vertices, target_faces = load_obj(fname)
target_vertices = normalize_vertices(target_vertices)
target_normals = calc_vertex_normals(target_vertices, target_faces)

# Render target depth maps (only depth channel)
target_depth = renderer.render(target_vertices, target_normals, target_faces)[..., 0]
save_images_new(target_depth, './out/target_depth/')

# Initialize mesh optimizer with a sphere
vertices, faces = make_sphere(level=2, radius=.5)
opt = MeshOptimizer(vertices, faces)
vertices = opt.vertices
snapshots = []

# Optimization loop on depth maps only
for i in tqdm(range(steps)):
    opt.zero_grad()
    normals = calc_vertex_normals(vertices, faces)
    images = renderer.render(vertices, normals, faces)[..., 0]  # keep only depth

    loss = (images - target_depth).abs().mean()
    

    loss.backward()
    opt.step()

    if show and i % snapshot_step == 0:
        snapshots.append(snapshot(opt))

    vertices, faces = opt.remesh()

# Save final optimized mesh with a timestamped name
timestamp = int(time.time())
save_obj(vertices, faces, f'./out/result_{timestamp}.obj')

# Save final depth map
save_images_new(images, './out/final_depth/')

if show:
    show(target_vertices, target_faces, snapshots)
