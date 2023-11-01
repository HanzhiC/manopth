import os
import os

import numpy as np
import torch
from torch.nn import Module

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth import rodrigues_layer, rotproj, rot6d
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)
import sys, os, time, math, pickle
os.environ["CC"] = "g++"

from handover.dex_ycb import DexYCB
from handover.config import get_config_from_args
from manopth.manolayer import ManoLayer
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import trimesh
from transformations import rotation_matrix

def vis_results_from_mano_layer(faces, verts, joints):
    verts = verts.view(-1, 3)
    verts = verts.cpu().numpy()
    verts /= 1000

    hand_mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    hand_mesh_o3d.compute_vertex_normals()
    return hand_mesh_o3d

def trimesh_to_o3d(mesh, scale=1., with_color=True):
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices * scale), 
                                     o3d.utility.Vector3iVector(mesh.faces))
    # mesh_o3d.paint_uniform_color([255 / 255, 229 / 255, 180 / 255])
    if with_color:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3]/255)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def draw_axis(T):
    axis_hand = trimesh.creation.axis(origin_size=0.01, transform=T, origin_color=[255,0,0], axis_radius=None, axis_length=None)
    return trimesh_to_o3d(axis_hand)

# create mano hand asset
cfg = get_config_from_args()
scene_data = DexYCB(cfg).get_scene_data(3)
name = scene_data["name"]
mano_sides = scene_data["mano_sides"]
betas = np.array(scene_data['mano_betas'])
device = torch.device('cuda:0')

asset_root = "/home/wiss/chenh/object_percept/handover-sim/handover/data/assets"
mano_asset_file = "{}_{}/mano.urdf".format(name.split('/')[0], mano_sides[0])
# asset_root = "/home/wiss/chenh/object_percept/dgrasp/rsc/"
# mano_asset_file = "mano/mano_mean.urdf"
_pose = scene_data['pose_m'][:, 0]
_sid = np.nonzero(np.any(_pose != 0, axis=1))[0][0]
_eid = np.nonzero(np.any(_pose != 0, axis=1))[0][-1]
pose = _pose.copy()
pose = torch.from_numpy(pose).float() #.to(device)
betas = torch.from_numpy(betas).float() #.to(device)
cmap = plt.get_cmap("turbo")


############################################################

mano_root = 'handover/data/mano_v1_2/models'
if mano_sides[0] == 'right':
    mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
else:
    mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

smpl_data = ready_arguments(mano_path)
hands_components = smpl_data['hands_components']
ncomps = 45
batch_size = 1
kintree_table = smpl_data['kintree_table']
parents = list(kintree_table[0].tolist())

# MANO parameters
th_betas = torch.Tensor(smpl_data['betas'].r).unsqueeze(0) # 1 x 10; 10 is the number of shape parameters
th_shapedirs = torch.Tensor(smpl_data['shapedirs'].r) # 778 x 3 x 10; 
th_posedirs = torch.Tensor(smpl_data['posedirs'].r) # 778 x 3 x 135 
th_v_template = torch.Tensor(smpl_data['v_template'].r) # 778 x 3
th_weights = torch.Tensor(smpl_data['weights'].r) # 778 x 16
th_J_regressor = torch.Tensor(smpl_data['J_regressor'].toarray()) # 16 x 778
th_faces = torch.Tensor(smpl_data['f'].astype(np.int32))[None]
th_betas = torch.Tensor(betas)

# hands mean 
hands_mean = smpl_data['hands_mean'] 
hands_mean = hands_mean.copy()
th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)  # 1 x 45
th_selected_comps = torch.Tensor(hands_components[:ncomps])  # 1 x 45

# step by step get mano mesh
th_trans = pose[_eid, :3][None] # [:, :3] is the global translation

# Solve for rotation of the hand
th_pose_coeffs = pose[_eid, 3:][None] # [:, 3:51] is the global rotation
th_hand_pose_coeffs = th_pose_coeffs[:, 3: 3 + ncomps] # [:, 3:48] is the hand local rotation coefficients, not the rotation!
th_full_hand_pose = th_hand_pose_coeffs.mm(th_selected_comps) # [:, 45] @ [45, 45] => [:, 45] hand local rotation
th_full_pose = torch.cat([th_pose_coeffs[:, :3], th_hands_mean+th_full_hand_pose], dim=1) # cat([:, 3], [:, 45]) Global+Locl

# Compute rotation matrices from the axis-angle with skipping global rotation
# th_pose_map is of shape [:, 16*9]; th_rot_map is of shape [:, 16*9]
# th_pose_map = th_rot_map - I (Why?); Because posedirs need the no identity matrix results
th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose) # [:, 48] => [:, 16*9]; ? while 16 insted of 21?
root_rot = th_rot_map[:, :9].view(batch_size, 3, 3) # [:, 9] => [:, 3, 3]
th_rot_map = th_rot_map[:, 9:]
th_pose_map = th_pose_map[:, 9:] # Hand local rotation

# Here we do mesh deformation!
# Full axis angle representation with root joint (! Joints position and vertices position in hand canonical space only depend on shape parameters)
th_v_shaped = th_v_template + torch.matmul(th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)  # [:, 778, 3] +  [778, 3, 10] @ [10, 1] => [:, 778, 3]
th_j = torch.matmul(th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1) # [:, 16, 778] @ [778, 3] => [:, 16, 3]
# Use the posture parameters to further deform meshes
th_v_posed = th_v_shaped + torch.matmul(th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1) # [:, 778, 3] + [778,3,135] @ [135, :]

# Compute global joint location, use the root joint location and rotation
root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1) # [:, 3, 1]
root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2)) # [:, 3, 3]|[:, 3, 1]=>[:, 4, 4]
all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3) # [:, 15*9] => [:, 15, 3, 3]

# Thumb, Index, Middle, Ring, Little, Skip finger tips!
lev1_idxs = [1, 4, 7, 10, 13]
lev2_idxs = [2, 5, 8, 11, 14]
lev3_idxs = [3, 6, 9, 12, 15]
lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]] # -1 because all rots doesn't have root rotation; R_root_joint
lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]] # -1 because all rots doesn't have root rotation; R_root_joint
lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]] # -1 because all rots doesn't have root rotation; R_root_joint
lev1_j = th_j[:, lev1_idxs]
lev2_j = th_j[:, lev2_idxs]
lev3_j = th_j[:, lev3_idxs]

# From base to tips
# Get lev1 results
all_transforms = [root_trans.unsqueeze(1)] # 
lev1_j_rel = lev1_j - root_j.transpose(1, 2) # t_root_joint [:, L, 3] - [:, 1, 3] => [:, L, 3]
lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)) # [:, 5, 3, 1][:, 5, 3, 3][:*5, 4, 4] T_root_joint
root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4) # [:, 5, 4, 4] => [:*5, 4, 4]
lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)  # [:*5, 4, 4] @ [:*5, 4, 4] => [:*5, 4, 4]; T_world_root @ T_root_joint
all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4)) # [:, 5, 4, 4] T_world_joint_lev1

# Get lev2 results
lev2_j_rel = lev2_j - lev1_j
lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

# Get lev3 results
lev3_j_rel = lev3_j - lev2_j
lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

## all_transforms contains T_world_joint for all 16 joints
reorder_idxs = [0, # root
                1, 6, 11, # thumb
                2, 7, 12, # index
                3, 8, 13, # center
                4, 9, 14, # ring
                5, 10, 15] # little
th_results = torch.cat(all_transforms, 1)[:, reorder_idxs] # [:, 16, 4, 4]
th_results_global = th_results # [:, 16, 4, 4]

# Bring the joint T_world_joint to each vertex so that we acquire the T_world_vertex finally
joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2) # [:, 16, 3] => [:, 16, 4]
tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3)) # [:, 16, 4, 4] @ [:, 16, 4, 1] => [:, 16, 4, 1]
th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1) # [:, 16, 4, 1] => [:, 16, 4, 4] => [:, 4, 4, 16] 
th_T = torch.matmul(th_results2, th_weights.transpose(0, 1)) # [:, 4, 4, 16] @ [778, 16] => [:, 4, 4, 778]
th_rest_shape_h = torch.cat([th_v_posed.transpose(2, 1), torch.ones((batch_size, 1, th_v_posed.shape[1]),  dtype=th_T.dtype, device=th_T.device), ], 1) # [:, 4, 778]
th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1) # [:, 4, 4, 778] * [:, 1, 4, 778] => [:, 4, 778] => [:, 778, 4]
th_verts = th_verts[:, :, :3]
th_jtr = th_results_global[:, :, :3, 3]

# Acquire tips 
if mano_sides[0] == 'right':  tips = th_verts[:, [745, 317, 444, 556, 673]]
else: tips = th_verts[:, [745, 317, 445, 556, 673]]
th_jtr = torch.cat([th_jtr, tips], 1)
th_jtr = th_jtr[:, [0, 
13, 14, 15, 16, 
1, 2, 3, 17, 
4, 5, 6, 18, 
10, 11, 12, 19, 
7, 8, 9, 20]]

# Final Translation
# th_jtr = th_jtr + th_trans.unsqueeze(1)
# th_verts = th_verts + th_trans.unsqueeze(1)

# DEBUG: VIS the hand mesh
mano_layer = ManoLayer(flat_hand_mean=False,
                        ncomps=45,
                        side=mano_sides[0], 
                        mano_root='handover/data/mano_v1_2/models',
                        use_pca=True)

with torch.no_grad():
    verts, joints = mano_layer(th_pose_coeffs=pose[_eid, 3:][None],  
    th_trans=pose[_eid, :3][None], th_betas=betas)
    mesh_correct = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts[0].cpu().numpy()/1000), 
    o3d.utility.Vector3iVector(th_faces[0].cpu().numpy())).paint_uniform_color([0, 1.0, 0])
    mesh_correct.compute_vertex_normals()
    vis = []
    for i in range(len(joints[0])):
        j = joints[0][i].cpu().numpy()
        j_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(j/1000)
        j_o3d.compute_vertex_normals()
        j_o3d.paint_uniform_color([1, 0, 0])
        # vis.append(j_o3d)
    
    for i in range(len(th_results_global[0, 0:])):
        # if 6 <= i < 9:
        T = th_results_global[0][i].cpu().numpy()
        T[:3, 3] += th_trans.squeeze().numpy()
        axis = draw_axis(T)
        vis.append(axis)
    o3d.visualization.draw([*vis, mesh_correct])
    

mesh_final = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(th_verts[0].cpu().numpy()), 
o3d.utility.Vector3iVector(th_faces[0].cpu().numpy())).paint_uniform_color([1, 0.1, 0])
mesh_final.compute_vertex_normals()

o3d.visualization.draw([mesh_final, mesh_correct])

# mesh_shaped = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(th_v_shaped[0].cpu().numpy()), 
# o3d.utility.Vector3iVector(th_faces[0].cpu().numpy())).paint_uniform_color([1, 0.1, 0])

# mesh_posed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(th_v_posed[0].cpu().numpy()), 
# o3d.utility.Vector3iVector(th_faces[0].cpu().numpy())).paint_uniform_color([0, 0.1, 0.929])

# mesh_temp = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(th_v_template.cpu().numpy()),
# o3d.utility.Vector3iVector(th_faces[0].cpu().numpy())).paint_uniform_color([0, 0.651, 0])

# mesh_shaped.compute_vertex_normals()
# mesh_posed.compute_vertex_normals()
# mesh_temp.compute_vertex_normals()

# o3d.visualization.draw([mesh_shaped, mesh_temp, mesh_posed])