from functools import partial
import matplotlib
import torch
import trimesh
from timeit import timeit
import matplotlib.pyplot as plt
import numpy as np
from core.remesh import calc_edge_length, calc_edges, calc_face_collapses, calc_face_normals, calc_vertex_normals, collapse_edges, flip_edges, pack, prepend_dummies, split_edges

def make_inputs(level):
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device='cuda', dtype=torch.float32)
    faces = torch.tensor(sphere.faces, device='cuda', dtype=torch.long)
    vertices,faces = prepend_dummies(vertices,faces)
    return vertices,faces

def fun(vertices,faces,collapse,split,flip):
    
    if collapse:
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        edge_length = calc_edge_length(vertices,edges) #E
        face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
        vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
        # random min_edgelen for benchmarking
        V = vertices.shape[0]
        min_edgelen = edge_length.mean() * (.9 + .11 * torch.rand(V,device=vertices.device))
        face_collapse = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edgelen,area_ratio=0.5)
        shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0) #e[0,1] 0...ok, 1...edgelen=0
        priority = face_collapse.float() + shortness
        vertices,faces = collapse_edges(vertices,faces,edges,priority)

    if split:
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        edge_length = calc_edge_length(vertices,edges) #E
        # replace with random mask for benchmarking
        splits = torch.randint(0,10,[edge_length.shape[0]],device=vertices.device)==0
        vertices,faces = split_edges(vertices,faces,edges,face_to_edge,splits,pack_faces=False)

    # pack
    vertices,faces = pack(vertices,faces)
    
    # flip (has work because of collapses and splits)
    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True) #E,2 F,3
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    return vertices,faces

inputs = make_inputs(4)
funs = {
    'collapse': partial(fun,collapse=True,split=False,flip=False),
    'split': partial(fun,collapse=False,split=True,flip=False),
    'collapse,split,flip': partial(fun,collapse=True,split=True,flip=True),
}

def test_fun(f,inputs,number):
    for _ in range(number):
        f(*inputs)
    torch.cuda.synchronize()

levels = np.arange(1,8)
number = 100

#warm-up
for f in funs.values():
    for level in levels:
        test_fun(f,make_inputs(level),number)

ts=[]
for f in funs.values():
    t=[]
    for level in levels:
        inputs = make_inputs(level)
        test_fun(f,inputs,number)
        torch.cuda.synchronize()
        t.append(timeit(lambda: test_fun(f,inputs,number),number=1)/number)            
    ts.append(t)

vertcounts = [make_inputs(level)[0].shape[0] for level in levels]

plt.plot(vertcounts, np.array(ts).T*1e3,'-o')
plt.legend(funs.keys())
plt.xscale('log')
plt.xlabel('vertices')
plt.ylabel('ms')
plt.grid()
plt.title(f'collapse_split_flip()')
plt.show()

def prepare_for_savefig():
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['lines.markersize'] = 4
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
    \usepackage{amsmath}"""
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.gca().set_facecolor('#ddd')
    plt.grid(color='w')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(which='both',length=0)

prepare_for_savefig()
t = np.array(ts)*1e3
t[2,:] -= t[0,:] + t[1,:]
plt.stackplot(vertcounts, t, labels=['collapse','split','flip'], colors=["#dddd54", "5454dd", "54dd54"]) 
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), loc='upper left')

plt.plot(vertcounts, np.cumsum(t,axis=0).T,'-ko')
plt.xscale('log')
plt.xlabel('vertices')
plt.ylabel('ms')
plt.grid(True)
plt.gcf().set_size_inches(4, 2)
plt.savefig(f"out/benchmark.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()

