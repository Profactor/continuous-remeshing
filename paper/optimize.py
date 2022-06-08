from dataclasses import dataclass, field
import time
from typing import Optional
import warnings
import torch
from tqdm import tqdm
from pathlib import Path
from core.opt import MeshOptimizer
from core.remesh import calc_edge_length, calc_edges, calc_vertex_normals
from util.func import laplacian, load_obj, make_sphere, make_star_cameras, normalize_vertices, save_images, to_numpy
from util.render import NormalsRenderer
from util.snapshot import Snapshot, snapshot
import numpy as np
try:
    from pyremesh import remesh_botsch
except:
    remesh_botsch = None

#suppress warning in torch.cartesian_prod()
warnings.filterwarnings("ignore",message='torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.')

@dataclass
class OptimizeSettings:
    #requires target fname or vertices/faces
    target_fname:Path = None
    target_vertices:torch.Tensor = None #V,3
    target_faces:torch.Tensor = None #F,3
    
    #requires steps or timeout
    steps:Optional[int] = 500
    timeout:Optional[float] = None

    outdir:str = 'out'
    method:str = 'ours' #adam,large,ours
    image_size:int = 512
    sphere_size:float = .5
    sphere_level:int = 2 #0->12,42,162,642,2562, 5->10k,40k,160k
    sphere_shift:tuple[float,float,float] = None
    cameras:tuple[int,int] = (4,4)
    device = 'cuda'

    #optimizer common
    lr:float = 0.5
    laplacian_weight:float = .1
    ramp:float = 3.
    betas:tuple[float,float,float] = (0.8,0.8,0)
    remesh_interval:int = 1
    edge_len_lims:tuple[float,float] = (.01,.15)
    
    #optimizer ours
    gammas:tuple[float,float,float] = (0,0,0)
    nu_ref:float = 0.3
    edge_len_tol:float = .5
    gain:float = .2
    local_edgelen:bool = True

    #optimizer adam remesh
    remesh_ratio:float = .5

    #result
    result_interval:int = 5
    result_meshes:bool = False
    result_snapshots:bool = False
    
    save_images:bool = False


@dataclass
class OptimizeResult:
    settings:OptimizeSettings
    target_vertices:torch.Tensor = None
    target_faces:torch.Tensor = None
    snapshots:list[Snapshot] = field(default_factory=list)
    

def make_optimizer(settings,vertices,faces):
    edges,_ = calc_edges(faces)
    mean_edge_length = calc_edge_length(vertices,edges).mean().item()
    lr = settings.lr * mean_edge_length
    Laplacian = None

    if settings.method=='adam':
        vertices.requires_grad_()
        opt = torch.optim.Adam([vertices],lr=lr,betas=settings.betas)
        edges,_ = calc_edges(faces)
        Laplacian = laplacian(vertices.shape[0],edges)
        loss = (vertices * (Laplacian@vertices)).mean() #warm-up
    elif settings.method=='ours':
        opt = MeshOptimizer(vertices,faces,lr=settings.lr,betas=settings.betas,gammas=settings.gammas,nu_ref=settings.nu_ref,
            edge_len_lims=settings.edge_len_lims,edge_len_tol=settings.edge_len_tol, gain=settings.gain, 
            laplacian_weight=settings.laplacian_weight, ramp=settings.ramp, 
            remesh_interval=settings.remesh_interval, local_edgelen=settings.local_edgelen)

        vertices = opt.vertices
    else:
        raise RuntimeError('unknown method')

    return opt,lr,vertices,Laplacian

def load_target_mesh(fname,device='cuda'):
    vertices,faces = load_obj(fname,device=device)
    vertices = normalize_vertices(vertices)
    return vertices,faces

def optimize(settings:OptimizeSettings):
    result = OptimizeResult(settings=settings)
    outdir = Path(settings.outdir)
        
    vertices,faces = make_sphere(level=settings.sphere_level,radius=settings.sphere_size,device=settings.device)
    if settings.sphere_shift:
        vertices += torch.tensor(settings.sphere_shift,device=settings.device)

    mv,proj = make_star_cameras(settings.cameras[0],settings.cameras[1],distance=10,
        image_size=[settings.image_size,settings.image_size],device=settings.device)

    renderer = NormalsRenderer(mv,proj,image_size=[settings.image_size,settings.image_size])
    
    if settings.target_vertices is None:
        target_vertices,target_faces =  load_target_mesh(settings.target_fname)
    else:
        target_vertices,target_faces = settings.target_vertices,settings.target_faces

    result.target_vertices,result.target_faces = target_vertices,target_faces

    target_normals = calc_vertex_normals(target_vertices,target_faces)
    target_images = renderer.render(target_vertices,target_normals,target_faces)

    if settings.save_images:
        save_images(target_images,outdir/'target_images')

    opt,lr,vertices,Laplacian = make_optimizer(settings,vertices,faces)
    start = time.time()
    step = 1
    last_remesh_step = 0
    with tqdm(desc='Optimize',total=settings.steps if settings.timeout is None else settings.timeout,leave=False) as tqdm_:
        is_last = False
        while not is_last:
            is_last = step==settings.steps if settings.steps else time.time()-start>settings.timeout

            opt.zero_grad()

            normals = calc_vertex_normals(vertices,faces)
            images = renderer.render(vertices,normals,faces)
            loss = (images-target_images).abs().mean()

            if isinstance(opt,torch.optim.Adam):
                #laplacian regularization
                loss = loss + (vertices * (Laplacian@vertices)).mean() * settings.laplacian_weight

            loss.backward()

            if isinstance(opt,torch.optim.Adam):
                #learning ramp
                ramped_lr = lr * min(1,(step-last_remesh_step) * (1-settings.betas[0]) / settings.ramp)
                opt.param_groups[0]['lr'] = ramped_lr

            opt.step()
            
            #snapshot
            with torch.no_grad():
                if (settings.result_interval and step % settings.result_interval == 1) or is_last:
                    if settings.method=='ours':
                        s = snapshot(opt)
                    else:
                        s = Snapshot(
                            step=step,
                            time=time.time()-start,
                            vertices=vertices.clone().requires_grad_(False),
                            faces=faces.clone(),
                        )
                    result.snapshots.append(s)

            #remesh
            if settings.remesh_interval is not None \
                and (step % settings.remesh_interval) == settings.remesh_interval-1 \
                and not is_last:

                if isinstance(opt,MeshOptimizer):
                    vertices,faces = opt.remesh()
                else:
                    with torch.no_grad():
                        edges,_ = calc_edges(faces)
                        mean_edge_length = calc_edge_length(vertices,edges).mean().item()
                        target_edgelen = mean_edge_length * settings.remesh_ratio
                        target_edgelen = max(target_edgelen, settings.edge_len_lims[0])
                        v = to_numpy(vertices).astype(np.double)
                        f = to_numpy(faces).astype(np.int32)
                        v,f = remesh_botsch(v,f,5,target_edgelen,True)
                        vertices = torch.tensor(v,dtype=torch.float,device=vertices.device).contiguous()
                        faces = torch.tensor(f,dtype=torch.long,device=vertices.device).contiguous()
                        opt,lr,vertices,Laplacian = make_optimizer(settings,vertices,faces)
                        last_remesh_step = step
                
                if vertices.shape[0]==0:
                    is_last = True #mesh collapsed

            if settings.save_images:
                save_images(images,outdir/'images')

            step += 1
            if settings.steps is not None:
                tqdm_.update(1)
            else:
                tqdm_.update(min(settings.timeout, round(time.time()-start,3)) - tqdm_.n)

    return result