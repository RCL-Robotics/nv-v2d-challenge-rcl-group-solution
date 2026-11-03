"""Optional regeneration of the local rendering proxy from the provided mesh.

This file is provenance for /work/tools/bat_render_mesh.npz. solve.py uses the
already generated proxy and does not need Open3D or trimesh at runtime.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

mesh = trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
# GLB contains per-triangle vertices. Welding is essential before decimation.
mesh.merge_vertices(merge_tex=True, merge_norm=True)
proxy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                 o3d.utility.Vector3iVector(mesh.faces))
proxy = proxy.simplify_quadric_decimation(6000)
proxy.remove_unreferenced_vertices()
Path('/work/tools').mkdir(parents=True, exist_ok=True)
np.savez('/work/tools/bat_render_mesh.npz', vertices=np.asarray(proxy.vertices),
         faces=np.asarray(proxy.triangles))
