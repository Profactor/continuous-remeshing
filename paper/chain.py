from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
import torch
from torch import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as tfunc
from tqdm import tqdm
from core.opt import lerp_unbiased

outdir = Path(f'out/chain_test')
drawing_dir = outdir/'drawing'
curves_dir = outdir/'curves'
drawing_dir.mkdir(parents=True,exist_ok=True)
curves_dir.mkdir(parents=True,exist_ok=True)

def interpolate(a,size):
    return tfunc.interpolate(a[None,None],size=size,mode='linear',align_corners=True)[0,0]
    
class ChainOptimizer:

    def __init__(self,x,edge_len_lims,lr=.25,betas=(.8,.95),nu_target=.3,edge_len_tolerance=.7,gain=.5):
        self._lr = lr
        self._betas = betas
        self._nu_target = nu_target
        self._edge_len_tolerance = edge_len_tolerance
        self._gain = gain
        self._m = torch.zeros_like(x)
        self._v = torch.tensor(0.)
        self._nu_mean = torch.tensor(0.)
        self._step = 0
        self._ref_len = torch.tensor(edge_len_lims[1])
        self._edge_len_lims = edge_len_lims

    @torch.no_grad()
    def step(self,x:torch.Tensor,z:torch.Tensor):
        self._step += 1
        lerp_unbiased(self._m,z.grad,self._betas[0],self._step)
        lerp_unbiased(self._v,z.grad.pow(2)[1:-1].mean(),self._betas[1],self._step)
        nu = self._m / (self._v.sqrt() + 1e-10)
        self._nu_mean = nu[1:-1].abs().mean()
        edge_len = x.diff().mean()
        z -= nu * self._lr * edge_len

        len_change = 1 + (self._nu_mean - self._nu_target) * self._gain
        self._ref_len *= len_change
        self._ref_len.clamp_(*self._edge_len_lims)
        
    @torch.no_grad()
    def remesh(self,x:torch.Tensor,z:torch.Tensor):
        edge_len = x.diff().mean()
        len_err = edge_len / self._ref_len - 1
        if len_err > self._edge_len_tolerance:
            size = size=2*len(x)-1
            x = interpolate(x,size)
            z = interpolate(z,size)
            z.requires_grad_()
            self._m = interpolate(self._m,size)

        return x,z

def target(x:torch.Tensor):
    k = pi / x[-1]
    fi = x * k
    return .05 * sin(fi) + .01 * sin(4*fi) + .005 * sin(13*fi)

def target_grad(x:torch.Tensor):
    k = pi / x[-1]
    fi = x * k
    return .05 * cos(fi) * k + .01 * cos(4*fi) * 4*k + .005 * cos(13*fi) * 13*k

@dataclass
class Hist:
    dist:list = field(default_factory=list)
    nu_mean:list = field(default_factory=list)
    ref_len:list = field(default_factory=list)
    edge_len:list = field(default_factory=list)
    x:list = field(default_factory=list)
    z:list = field(default_factory=list)

def run(min_verts, max_verts, steps):
    x = torch.linspace(0,.1,min_verts)
    z = torch.zeros_like(x)
    z.requires_grad_()
    l = x.diff().mean().item()

    opt = ChainOptimizer(x, edge_len_lims=(l*(min_verts-1)/(max_verts-1),l))

    hist = Hist()
    
    for _ in range(steps):
        x_fine = interpolate(x,512)
        z_fine = interpolate(z,512)
        dz_fine = torch.gradient(z_fine,spacing=[x_fine])[0]
        loss = (dz_fine - target_grad(x_fine)).pow(2).mean()

        z.grad = None
        loss.backward()
        opt.step(x,z)

        with torch.no_grad():
            z[0] = z[-1] = 0

        target_z_fine = target(x_fine)
        dist = (z_fine-target_z_fine).abs().max() / target_z_fine.max()
        hist.dist.append(dist.item())
        hist.nu_mean.append(opt._nu_mean.item())
        hist.ref_len.append(opt._ref_len.item())
        hist.edge_len.append(x.diff().mean().item())
        hist.x.append(x.clone())
        hist.z.append(z.detach().clone())

        x,z = opt.remesh(x,z)

    return opt,hist

def prepare_plot():
    ax = plt.gca()
    ax.set_facecolor('#ddd')
    ax.grid(color='w')
    ax.minorticks_off()
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

iterations = 180
opt,hist = run(5,65,iterations)

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['grid.linewidth'] = 0.2
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x_t = torch.linspace(0,hist.x[0][-1],100)

pdf_it = 20

for it in tqdm(range(0,iterations)):
    fig = plt.figure(figsize=(4,3))
    x_t = torch.linspace(0,hist.x[0][-1],100)
    plt.fill(x_t,target(x_t), color='#ffff54ff')
    plt.plot(x_t,target(x_t), '--', color='gray')
    plt.plot(hist.x[it], hist.z[it], '-ok', linewidth=2, markerfacecolor='w')
    plt.xlim(-.003, .103)
    plt.ylim(-.01, .08)
    plt.axis('off')

    if it==pdf_it:
        fig.set_size_inches(4,1.5)
        plt.savefig(outdir/f"chain_{it}.pdf", format='pdf')

    fig.set_size_inches(4,3)
    plt.text(hist.x[0][-1]/2,.07,f'$t={it}, l={hist.edge_len[it]:.3f}$',horizontalalignment='center')
    plt.savefig(drawing_dir/f"chain_drawing_{it:03d}.png", dpi=600)
    plt.close(fig)

    fig = plt.figure(figsize=(4,3))
    plt.plot(hist.nu_mean[:it], '-r', label=r'relative velocity $\nu$')
    plt.axhline(y = opt._nu_target, color = 'r', linestyle = '--', label=r'$\nu_{ref}=0.3$')
    
    tlen = np.array(hist.ref_len[:it])
    steps = np.arange(0,len(hist.dist[:it]))
    poly_x = np.concatenate((steps,steps[::-1]))
    poly_y = np.concatenate((tlen*(1-opt._edge_len_tolerance),tlen[::-1]*(1+opt._edge_len_tolerance)))
    plt.fill(poly_x,poly_y, color='lightgreen')
    
    plt.plot(steps, tlen, '--g', label=r'target edge len $l_{ref}$')
    edge_len_line, = plt.plot(steps, hist.edge_len[:it], '-k', label=r'edge len $l$')
    plt.xlim(-3,iterations+3)
    plt.ylim(1.5e-4,1.1)
    plt.yscale('log')
    plt.xlabel('step')
    plt.legend(loc='upper right',framealpha=1)
    prepare_plot()
    plt.savefig(curves_dir/f"chain_curves_{it:03d}.png", dpi=600)
    
    if it==iterations-1:
        plt.ylabel(r'$\nu,l$')
        plt.savefig(outdir/f"chain_curves.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)

    plt.close(fig)

fig = plt.figure(figsize=(4,3))

for i,n in enumerate([5,9,17,33,65]):
    opt,hist = run(n,n,iterations)
    dashes = [[],[12],[6],[3,4],[2,3]][i]
    plt.plot(hist.dist, dashes=dashes, color='gray',label=f'{n} vertices')

opt,hist = run(5,64,iterations)

plt.plot(hist.dist,color='darkgreen',linewidth=1.5,label='with remeshing')
plt.xlabel('step')
plt.ylabel('max. normalized distance')
plt.yscale('log')
plt.legend()
prepare_plot()

plt.savefig(outdir/f"chain_distance.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)

