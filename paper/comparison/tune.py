from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from paper.optimize import OptimizeResult, OptimizeSettings, load_target_mesh, optimize
import numpy as np
from paper.comparison.settings import method_settings
from util.func import to_numpy
from util.igl import igl_distance

@dataclass
class TuneResult:
    tunestep:int
    settings:OptimizeSettings
    rms_distance:float
    
datadir = Path('data')
outdir = Path('out/comparison')
outdir.mkdir(parents=True,exist_ok=True)

models = 'bunny','lucy','armadillo','nefertiti','horse'

target_meshes = {model:load_target_mesh(datadir/model) for model in models}
target_meshes_np = {model:to_numpy(*target_meshes[model]) for model in models}

def warmup():
    settings = deepcopy(method_settings['adam'])
    settings.target_vertices,settings.target_faces = target_meshes[models[0]]
    settings.timeout = 2
    settings.steps = None
    optimize(settings)
    
warmup()
tunesteps = 500
methods = method_settings.keys()
    
for method in (method_bar:=tqdm(methods)):
    method_bar.set_description(method)
    settings = deepcopy(method_settings[method])
    settings.timeout = 3
    settings.steps = None
    settings.result_interval = 0
    settings.result_meshes = True

    hyperparams = ['lr','betas']
    if method == 'ours':
        hyperparams.extend(['laplacian_weight','nu_ref','edge_len_lims','gain','ramp'])
    if method == 'adam':
        hyperparams.extend(['laplacian_weight','ramp'])
    if method == 'adam_remesh':
        hyperparams.extend(['laplacian_weight','ramp','remesh_interval','remesh_ratio','edge_len_lims'])

    all_tune:list[TuneResult] = []
    best_tune:list[TuneResult] = []

    for tunestep in tqdm(range(tunesteps),desc='Tunestep',leave=False):
        if best_tune:
            # modify a random hyperparameter
            settings = deepcopy(best_tune[-1].settings)
            name = random.choice(hyperparams)
            value = settings.__getattribute__(name)
            ind = random.randrange(0,len(value)) if hasattr(value,'__len__') else None
            elem = value[ind] if ind is not None else value
            elem = type(elem)(elem * 2 ** random.uniform(-1,1)) 
            if ind is None:
                value = elem
            else:
                value = list(value)
                value[ind] = elem
            settings.__setattr__(name,value)

            settings.edge_len_lims = list(settings.edge_len_lims)
            settings.edge_len_lims[0] = max(settings.edge_len_lims[0], 0.004)
            settings.edge_len_lims[1] = max(settings.edge_len_lims[0], settings.edge_len_lims[1])
            
            settings.betas = [min(beta,.999) for beta in settings.betas]
                
        results:list[OptimizeResult] = []
        for model in (model_bar:=tqdm(models,leave=False)):
            model_bar.set_description(model) 
            settings.target_vertices,settings.target_faces = deepcopy(target_meshes[model])
            result = optimize(settings)
            snapshot = result.snapshots[-1]
            if snapshot.vertices.shape[0] == 0 or snapshot.vertices.shape[0] > 1e6:
                break
            vertices_np,faces_np = to_numpy(snapshot.vertices,snapshot.faces)
            vertex_distance,rms_distance,max_distance = igl_distance(vertices_np,faces_np,*target_meshes_np[model])
            results.append((result,rms_distance))

        if len(results)!=len(models):
            continue        

        rms_distance = np.array([d for r,d in results])
        rms_distance = np.mean(rms_distance**2)**.5
        
        tune_result = TuneResult(tunestep=tunestep,settings=settings,rms_distance=rms_distance)
        all_tune.append(tune_result)
        if (not best_tune) or rms_distance < best_tune[-1].rms_distance:
            best_tune.append(tune_result)

        if best_tune:
            with open(outdir/f'settings_{method}.txt', 'w') as file:
                print(f'rms = {best_tune[-1].rms_distance}',file=file)
                params = ', '.join([f'{name}={best_tune[-1].settings.__getattribute__(name)}' for name in hyperparams])
                print(f'params = {params}',file=file)

            plt.clf()
            for name in hyperparams:
                if hasattr(settings.__getattribute__(name),'__len__'):
                    for j in range(len(settings.__getattribute__(name))):
                        all, = plt.plot([t.settings.__getattribute__(name)[j] for t in all_tune],'--')
                        best, = plt.plot([t.tunestep for t in best_tune],[t.settings.__getattribute__(name)[j] for t in best_tune],'-',color=all.get_color(),label=f'{name}[{j}]')
                else:
                    all, = plt.plot([t.settings.__getattribute__(name) for t in all_tune],'--')
                    best, = plt.plot([t.tunestep for t in best_tune],[t.settings.__getattribute__(name) for t in best_tune],'-',color=all.get_color(),label=name)

            plt.plot([t.rms_distance for t in all_tune],'--k')
            plt.plot([t.tunestep for t in best_tune],[t.rms_distance for t in best_tune],'-k',label='rms distance')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=[1.05, 1], loc='upper left')
            plt.savefig(outdir/f'tuning_{method}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
