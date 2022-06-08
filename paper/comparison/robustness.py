from copy import deepcopy
import pickle
from paper.comparison.settings import models,method_settings
from tqdm import tqdm
import numpy as np
from pathlib import Path
from paper.optimize import load_target_mesh, optimize
from util.func import to_numpy
from util.igl import igl_distance
import matplotlib
import matplotlib.pyplot as plt
import re

def set_value(settings,name,value):
    match = re.match(r'(.+)\[(.+)\]',name)
    if not match:
        settings.__setattr__(name,value)
    else:
        name = match.group(1)
        ind = int(match.group(2))
        values = list(settings.__getattribute__(name))
        values[ind] = value
        settings.__setattr__(name,values)

fname = 'out/robustness/results.pickle'

if not fname:
    datadir = Path('data')
    model_names = 'nefertiti','bunny','armadillo','lucy','horse'
    method = 'ours'
    setting_values = {
        #'lr': np.logspace(-2,1,3),
        #'laplacian_weight': np.logspace(-3,0,10),
        'target_speed': np.logspace(-1,0,30),
        'edge_len_gain': np.logspace(-2,1,60),
        #'remesh_interval': np.unique(np.logspace(0,2,10).astype(int)),
    }

    target_meshes = {model:load_target_mesh(datadir/model) for model in models}
    target_meshes_np = {model:to_numpy(*target_meshes[model]) for model in models}

    results={}
    for setting_name in (setting_name_bar:=tqdm(setting_values.keys(),leave=True)):
        setting_name_bar.set_description(setting_name)
        rms = []
        for setting_value in tqdm(setting_values[setting_name],leave=False,desc='setting_value'):
            setting_rms = []
            for model in (model_bar:=tqdm(model_names,leave=False)):
                model_bar.set_description(model)
                settings = deepcopy(method_settings[method])
                settings.target_vertices, settings.target_faces = target_meshes[model]
                settings.timeout = 3
                settings.steps = None
                settings.result_interval = 0
                set_value(settings,setting_name,setting_value)
                result = optimize(settings)
                if not result.snapshots:
                    break
                snapshot = result.snapshots[-1]
                if snapshot.vertices.shape[0] == 0 or snapshot.vertices.shape[0] > 1e6:
                    break
                vertices_np,faces_np = to_numpy(snapshot.vertices,snapshot.faces)
                vertex_distance,rms_distance,max_distance = igl_distance(vertices_np,faces_np,*target_meshes_np[model])
                setting_rms.append(rms_distance)
            rms.append((np.array(setting_rms)**2).mean()**.5 if setting_rms else None)
        results[setting_name] = (setting_values[setting_name],rms)
            #plt.plot(setting_values[setting_name],rms,label=setting_name)

    outdir = Path('out/robustness')
    outdir.mkdir(parents=True,exist_ok=True)
    fname = outdir/'results.pickle'
    with open(fname, 'wb') as file:
        pickle.dump(results,file)

plt.figure(figsize=(4,1.5))
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# matplotlib.rcParams['xtick.major.pad']='300'
# matplotlib.rcParams['ytick.major.pad']='300'

plt.gca().set_facecolor('#ddd')
plt.grid(color='w')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
#plt.tick_params(axis='y',which='both',length=0)

with open(fname, 'rb') as file:
    results = pickle.load(file)

labels = {
    'target_speed': r'Reference Velocity $\nu_{ref}$',
    'edge_len_gain': r'Gain $G$',
    'remesh_interval': r'Remesh Interval',
    'betas[0]': r'$\beta_1$',
    'betas[1]': r'$\beta_2$',
    'betas[2]': r'$\beta_3$',
    'gammas[1]': r'$\gamma_1$',
    'gammas[2]': r'$\gamma_2$',
}
for setting_name,(values,rms) in results.items():
    label = labels[setting_name] if setting_name in labels else setting_name
    plt.plot(values,rms,label=label)

#plt.xlim(.04,30)
#plt.ylim(1e-4,3e-1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Parameter Value')
plt.ylabel('RMS Distance')
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(f'out/robustness.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)