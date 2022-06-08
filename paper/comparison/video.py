from pathlib import Path
import torch
from tqdm import tqdm
from core.opt import MeshOptimizer
from core.remesh import calc_edge_length, calc_edges
from paper.optimize import optimize
from util.func import save_obj, save_ply
from util.view import show
from paper.blender_render import render_folder
from paper.comparison.settings import method_settings

model = 'lucy'
method = 'ours'
settings = method_settings[method]
settings.target_fname = f'data/{model}.obj'
settings.steps = 200
#settings.cameras = (5,5)
#settings.edge_len_lims = (0.005,0.15)
settings.result_interval = 5

outdir = Path(f'out/video/{model}_{method}')
meshdir = outdir/'mesh'
meshdir.mkdir(parents=True,exist_ok=True)

if True:
    result = optimize(settings)
    save_obj(result.target_vertices,result.target_faces,outdir/'target')
    show(result.target_vertices,result.target_faces,result.snapshots)

    vertex_count = None
    for i,s in enumerate(tqdm(result.snapshots)):
        fname = meshdir/f'{model}_{s.step:05}_{int(s.time*1e3):06}ms.ply'
        if isinstance(s.optimizer,MeshOptimizer):
            edgelen = s.optimizer._ref_len
            edgelen = torch.stack((edgelen,edgelen,edgelen),dim=-1)
        else:
            if s.vertices.shape[0] != vertex_count:
                vertex_count = s.vertices.shape[0]
                edges,_ = calc_edges(s.faces)
                mean_edge_length = calc_edge_length(s.vertices,edges).mean().item()
                edgelen = torch.full_like(s.vertices,fill_value=mean_edge_length)
        clim = .03
        vc = (edgelen / clim).clamp(0,1)
        save_ply(fname,s.vertices,s.faces,vc)

render_folder(meshdir,material='Jet',thickness=0.001,camera='auto',target=outdir/'target.obj')
#render_folder(meshdir,material='Jet',thickness=0.0005,camera='cam_smilodon_detail',target=outdir/'target.obj')