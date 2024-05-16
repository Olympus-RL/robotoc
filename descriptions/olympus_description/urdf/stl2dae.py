import os

import pymeshlab;

ms = pymeshlab.MeshSet()
meshes = []
folder_path = '/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/meshes/'

for file in os.listdir(folder_path):
    if file.endswith('.STL'):
        stl_file = os.path.join(folder_path, file)
        meshes.append(stl_file)

for stl_file in meshes:
    dae_file = stl_file.replace('.STL', '.dae')
    ms.load_new_mesh(stl_file)
    ms.save_current_mesh(dae_file)