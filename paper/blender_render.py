import subprocess
from pathlib import Path

BLENDER_EXEC = r"C:\Program Files\Blender Foundation\Blender 2.93\blender.exe"

def render_folder(folder:str, material:str, thickness:float, camera:str, target:str=None):
    """run blender to render all meshes in a folder"""
    args = [BLENDER_EXEC, 
        "-b" , "data/template.blend", 
        "--python", "paper/blender.py", 
        "--", 
        "--folder", str(Path(folder).resolve()),
        "--material", material,
        "--thickness", str(thickness),
        "--camera", camera,
        ]
    if target is not None:
        args.extend(["--target", str(Path(target).resolve())])

    subprocess.run(args, check=True)
