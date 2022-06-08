from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
from paper.comparison.settings import pretty_method, method_colors

outdir = Path('out/comparison')
meshdir = outdir/'mesh'

with open(outdir/'results.pickle', 'rb') as file:
    results = pickle.load(file)

methods = 'adam','adam_remesh','adam_remesh_complex','ours'
models = results.keys()
#models = 'bunny','lucy','armadillo','planck','nefertiti','horse','deer','smilodon'
spec = {
    'ours': '-o',
    'adam': '-o',
    'adam_remesh': '-o',
    'adam_remesh_complex': '--^',
}

matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['grid.linewidth'] = 0.2
matplotlib.rcParams['lines.linewidth'] = .75
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['ytick.major.pad'] = 3
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig = plt.figure(figsize=(4,2))
ax = fig.gca()
pretty_models = [model.capitalize() for model in models] 

for method_ind,method in enumerate(methods):
    rms = [results[model][method]['rms_distances'][-1] for model in models]
    plt.plot(rms,spec[method],label=pretty_method[method],markersize=2,color=method_colors[method])
plt.axvline(x=4.5, color='k')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(pretty_models)
plt.ylabel('RMS Distance')
plt.yscale('log')
plt.legend()
ax.set_facecolor('#ddd')
ax.grid(color='w')
ax.minorticks_off()
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(outdir/'comparison_overview.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
