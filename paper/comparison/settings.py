from paper.optimize import OptimizeSettings

models = 'nefertiti','bunny','armadillo','lucy','horse','deer','smilodon'

method_settings = {
    'ours': OptimizeSettings(lr=0.37,betas=(0.8,0.8,0),laplacian_weight=0.013,gammas=(0,0,0),nu_ref=0.3,edge_len_lims=(0.01,0.15),gain=0.2,ramp=1.5), #rms=0.00145
    'adam': OptimizeSettings(lr=0.33,betas=(0.81,0.81),laplacian_weight=8.4,ramp=2.5), #rms=0.00366
    'adam_remesh': OptimizeSettings(lr=0.205,betas=(0.812,0.99),laplacian_weight=0.859,ramp=10.7,remesh_interval=64,remesh_ratio=0.413, edge_len_lims=(.005,.2)), #rms=0.00151
    'adam_remesh_complex': OptimizeSettings(lr=0.41, betas=[0.87, 0.96], laplacian_weight=0.30, ramp=5, remesh_interval=100, remesh_ratio=0.45, edge_len_lims=[0.004, 0.2])
}

for method,settings in method_settings.items():
    settings.method = method.split('_')[0]
    if method == 'adam':
        settings.remesh_interval = None
        settings.sphere_level = 5

pretty_method = {
    'ours': 'Ours',
    'adam': 'Adam',
    'adam_remesh': 'Adam-Remesh',
    'adam_remesh_complex': 'Adam-Remesh Complex',
}

method_colors = {
    'ours': 'g',
    'adam': 'r',
    'adam_remesh': 'm',
    'adam_remesh_complex': 'm',
}