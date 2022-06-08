from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
from paper.comparison.settings import models, pretty_method, method_colors

outdir = Path('out/comparison')
meshdir = outdir/'mesh'

with open(outdir/'results.pickle', 'rb') as file:
    results = pickle.load(file)

methods = 'adam','adam_remesh','adam_remesh_complex','ours'
names = results.keys()

linestyles = {
    'ours': '-',
    'adam': '-',
    'adam_remesh': '-',
    'adam_remesh_complex': '--',
}

matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['legend.fontsize'] = 5
matplotlib.rcParams['xtick.labelsize'] = 3
matplotlib.rcParams['ytick.labelsize'] = 3
matplotlib.rcParams['grid.linewidth'] = 0.2
matplotlib.rcParams['lines.linewidth'] = .75
matplotlib.rcParams['xtick.major.pad'] = 1.5
matplotlib.rcParams['ytick.major.pad'] = 1.5
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

w_img = .12
w_plot = .17
w_plot_inner = .135
h_plot_inner = .115

fig = plt.figure(figsize=(4,4))
pretty_names = [name.capitalize() for name in names] 
for model_ind,model in enumerate(names):

    ax_img = fig.add_axes((0,-model_ind*w_img,w_img,w_img))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylabel(pretty_names[model_ind])

    ax_img.imshow(plt.imread(outdir/f'target_mesh/render/{model}.png'),aspect='equal')
    x0 = w_img + w_plot - w_plot_inner
    ax_dist = fig.add_axes((x0,-model_ind*w_img,w_plot_inner,h_plot_inner))
    ax_dist.set_ylim(3e-4,3e-1)
    if model_ind==0:
        ax_dist.set_title('Distance')

    ax_edgelen = fig.add_axes((x0+w_plot,-model_ind*w_img,w_plot_inner,h_plot_inner))
    ax_edgelen.set_ylim(8e-3,3e-1)
    if model_ind==0:
        ax_edgelen.set_title('Edge Length')

    ax_flips = fig.add_axes((x0+2*w_plot,-model_ind*w_img,w_plot_inner,h_plot_inner))
    ax_flips.set_ylim(1e-4,6e-1)
    if model_ind==0:
        ax_flips.set_title('Flip Ratio')

    for ax in [ax_dist,ax_edgelen,ax_flips]:
        ax.set_xlim(0,3)
        ax.set_yscale('log')
        ax.set_facecolor('#ddd')
        ax.grid(color='w')
        if model_ind==0:
            ax.xaxis.set_ticks_position('top')
        if model_ind>-1 and model_ind<len(names)-1:
            ax.xaxis.set_ticklabels([])
        if model_ind==len(names)-1:
            ax.set_xlabel('time [s]')

    for method_ind,method in enumerate(methods):
        result = results[model][method]
        ax_dist.plot(result['times'], result['rms_distances'],color=method_colors[method], linestyle=linestyles[method],label=pretty_method[method])
        ax_edgelen.plot(result['times'], result['edge_lengths'],color=method_colors[method], linestyle=linestyles[method],label=pretty_method[method])
        ax_flips.plot(result['times'], result['flip_ratios'],color=method_colors[method], linestyle=linestyles[method],label=pretty_method[method])

    if model_ind==len(names)-1:
        ax_flips.legend(bbox_to_anchor=(0,-.7))

for ax in fig.axes:
    ax.minorticks_off()
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.savefig(outdir/'comparison_curves.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
