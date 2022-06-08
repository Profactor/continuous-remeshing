import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(4,3))#,layout='tight')
img = plt.imread('data/smilodon_00376_011810ms.png')

matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['axes.linewidth'] = 0.2
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['ytick.major.width'] = .2

ax = fig.add_axes([0, 0, 0.75, 1])
plt.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
    
cb = fig.add_axes([.8, .1, 0.03, .8])
clim = .03
v = np.arange(0,.6,.0001)
cmap = np.stack((v,np.ones_like(v),np.ones_like(v))).T
cmap = matplotlib.colors.hsv_to_rgb(cmap)
cmap = cmap[:,None,:]
cb.imshow(cmap,extent=[0,1,0,clim])
cb.set_aspect('auto')
cb.set_xticks([])
cb.set_yticks(np.arange(0,clim+1e-3,.01))
cb.yaxis.tick_right()

plt.savefig(f'out/rib.pdf', format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.03)
