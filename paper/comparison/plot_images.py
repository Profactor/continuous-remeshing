from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

outdir = Path('out/comparison')
meshdir = outdir/'mesh'

model_names = ('bunny','lucy'),('deer','smilodon')

pretty_names = {
    'adam': 'Adam',
    'adam_remesh': 'Adam-Remesh',
    'adam_remesh_complex': 'Adam-Remesh C',
    'ours': 'Ours',
}

method_count = len(pretty_names.items())

matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['xtick.major.width'] = .2
matplotlib.rcParams['axes.linewidth'] = 0.2
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

w_img = .1
fig = plt.figure(figsize=(8,8))
for model_row_ind,model_row in enumerate(model_names):
    for model_col_ind,model in enumerate(model_row):
        for method_ind,method in enumerate(pretty_names.keys()):
            for time_ind,time in enumerate([1,2,3]):
                x = model_col_ind * (method_count+.2) + method_ind
                y = model_row_ind * 3.5 + time_ind
                ax_img = fig.add_axes((x*w_img,-y*w_img,w_img,w_img))
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                if time_ind==0:
                    ax_img.set_title(pretty_names[method])
                if method_ind==0:
                    ax_img.set_ylabel(f't = {time}s')
                ax_img.imshow(plt.imread(outdir/f'mesh/render/{model}_{method}_{time}s.png'),aspect='equal')

for ax in fig.axes:
    ax.set_facecolor('w')            
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

w = np.max([ax.get_position().max[0] for ax in fig.axes])
h = np.min([ax.get_position().min[1] for ax in fig.axes])
cb = fig.add_axes([0.2 * w,h-.03, 0.6 * w, 0.007])
clim = [1e-3,3e-3]
v = np.arange(0,2,.01)
cmap = np.stack((v,2-v,np.zeros(v.shape)),axis=-1).clip(min=0,max=1)[None]
cb.imshow(cmap,extent=[clim[0],clim[1],0,1])
cb.set_aspect('auto')
cb.set_yticks([])
cb.ticklabel_format(style='sci',scilimits=(0,0))

fig.savefig(outdir/'comparison_images.pdf', format='pdf', dpi=900, bbox_inches='tight', pad_inches=0.03)