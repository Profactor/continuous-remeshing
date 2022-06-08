from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from util.func import load_obj, make_sphere,make_star_cameras, normalize_vertices, save_obj, save_images
from util.render import NormalsRenderer
from tqdm import tqdm
from util.snapshot import snapshot
try:
    from util.view import show
except:
    show = None

fname = 'data/lucy.obj'
steps = 100
snapshot_step = 1

mv,proj = make_star_cameras(4,4)
renderer = NormalsRenderer(mv,proj,[512,512])

target_vertices,target_faces =  load_obj(fname)
target_vertices = normalize_vertices(target_vertices)
target_normals = calc_vertex_normals(target_vertices,target_faces)
target_images = renderer.render(target_vertices,target_normals,target_faces)
save_images(target_images[...,:3], './out/target_images/')
save_images(target_images[...,3:], './out/target_alpha/')

vertices,faces = make_sphere(level=2,radius=.5)

opt = MeshOptimizer(vertices,faces,local_edgelen=False)
vertices = opt.vertices
snapshots = []

for i in tqdm(range(steps)):
    opt.zero_grad()
    normals = calc_vertex_normals(vertices,faces)
    images = renderer.render(vertices,normals,faces)
    loss = (images-target_images).abs().mean()
    loss.backward()
    opt.step()

    if show and i%snapshot_step==0:
        snapshots.append(snapshot(opt))

    vertices,faces = opt.remesh()

save_obj(vertices,faces,'./out/result.obj')
save_images(images[...,:3], './out/images/')
save_images(images[...,3:], './out/alpha/')

if show:
    show(target_vertices,target_faces,snapshots)