import re
import bpy
import os

import argparse
import sys
import numpy as np
from pathlib import Path

sys.argv.remove('--')

parser = argparse.ArgumentParser(description="Render OBJ meshes from a given viewpoint, with wireframe.")
parser.add_argument("--folder", required=True, type=Path)
parser.add_argument("--material", required=True, type=str)
parser.add_argument("--thickness", required=True, type=float)
parser.add_argument("--camera", required=True, type=str)
parser.add_argument("--target", required=False, type=Path)
params = parser.parse_known_args()[0]

outdir = params.folder/'render'
outdir.mkdir(parents=True,exist_ok=True)

bpy.ops.object.select_all(action='DESELECT')

def import_mesh(filename, collection_index):
    if filename.suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=str(filename))
    elif filename.suffix == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(filename))
    else:
        return None

    obj = bpy.context.selected_objects[-1]
    bpy.context.view_layer.objects.active = obj
    if filename.suffix == ".ply":
        obj.rotation_euler[0] = np.pi / 2

    bpy.ops.object.move_to_collection(collection_index=collection_index)
    return obj

if params.target is not None:
    import_mesh(params.target,collection_index=1)

for filename in params.folder.iterdir():
    obj = import_mesh(filename,collection_index=2)
    if obj is None:
        continue

    obj.data.materials.clear()
    obj.data.materials.append(bpy.data.materials[params.material])
    obj.data.materials.append(bpy.data.materials['Wireframe'])
    
    bpy.ops.object.shade_smooth()

    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj.modifiers["Wireframe"].use_replace = False
    obj.modifiers["Wireframe"].use_even_offset = False
    obj.modifiers["Wireframe"].material_offset = 1
    obj.modifiers["Wireframe"].thickness = params.thickness

    if params.camera=='auto':
        camera = f'cam_{filename.stem.split("_")[0]}'
    else:
        camera = params.camera
    
    if not camera in bpy.data.objects.keys():
        camera = 'cam_default'
        
    bpy.context.scene.camera = bpy.data.objects[camera]
    bpy.context.scene.render.filepath = str((outdir/filename.stem).with_suffix('.png'))
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
