from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
from core.remesh import calc_edge_length, calc_edges
from paper.optimize import load_target_mesh, optimize
from paper.blender_render import render_folder
from paper.comparison.settings import models,method_settings
from util.func import save_obj, save_ply, to_numpy
from util.igl import igl_distance, igl_flips
from util.view import show
from tqdm import tqdm

datadir = Path('data')
outdir = Path('out/comparison')
target_meshdir = outdir/'target_mesh'
target_meshdir.mkdir(parents=True,exist_ok=True)
meshdir = outdir/'mesh'
meshdir.mkdir(parents=True,exist_ok=True)

results = {}
methods = method_settings.keys()

for model in (model_bar:=tqdm(models)):
    model_bar.set_description(model) 
    target_vertices,target_faces = load_target_mesh(datadir/model)
    target_vertices_np,target_faces_np = to_numpy(target_vertices,target_faces)
        
    results[model] = {} 

    save_obj(target_vertices,target_faces,target_meshdir/model)
    
    for method in (method_bar:=tqdm(methods,leave=False)):
        method_bar.set_description(method)
        settings = deepcopy(method_settings[method])
        settings.target_vertices,settings.target_faces = target_vertices,target_faces
        settings.steps = None
        settings.timeout = 3
        settings.result_interval = 5
        result = optimize(settings)

        times = []
        vertex_distances = []
        rms_distances = []
        edge_lengths = []
        flip_ratios = []

        for snapshot in tqdm(result.snapshots,leave=False,desc='Distance'):
            times.append(snapshot.time)
            vertices_np,faces_np = to_numpy(snapshot.vertices,snapshot.faces)
            vertex_distance,rms_distance,max_distance = igl_distance(vertices_np,faces_np,target_vertices_np,target_faces_np)
            vertex_distances.append(vertex_distance)
            rms_distances.append(rms_distance)
            flip, flip_ratio = igl_flips(vertices_np,faces_np,target_vertices_np,target_faces_np)    
            flip_ratios.append(flip_ratio)
            edges,_ = calc_edges(snapshot.faces)
            edge_lengths.append(calc_edge_length(snapshot.vertices,edges).mean().item())
        
        for time in 1,2,3:
            if time<settings.timeout:
                ind = np.nonzero([s.time>=time for s in result.snapshots])[0][0]
            else:
                ind = -1
            snapshot = result.snapshots[ind]
            clim = [1e-3,3e-3]
            vc = ( (vertex_distances[ind]-clim[0]) / (clim[1]-clim[0]) ).clip(min=0,max=1)
            vc = np.stack((vc,vc,vc),axis=-1)
            fname = meshdir/f'{model}_{method}_{time}s'
            save_ply(fname, snapshot.vertices, snapshot.faces, vc)
            
        results[model][method] = dict(times=times,rms_distances=rms_distances,edge_lengths=edge_lengths,flip_ratios=flip_ratios)

with open(outdir/'results.pickle', 'wb') as file:
    pickle.dump(results,file)
    
render_folder(meshdir,material='RedGreen',thickness=0.001,camera='auto')
render_folder(target_meshdir,material='Gray',thickness=0.00,camera='auto')


