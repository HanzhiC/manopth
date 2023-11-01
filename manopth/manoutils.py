import numpy as np
import torch

def compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_finger):
    T_world_joints_finger = torch.cat([T_world_joints_root, T_world_joints_finger], dim=1) # [:, 1+3, 4, 4]
    T_world_parentjoints = T_world_joints_finger[:, :-1]
    T_world_childjoints  = T_world_joints_finger[:, 1:]
    T_parent_child = torch.matmul(torch.inverse(T_world_parentjoints), T_world_childjoints)
    return T_parent_child

def compute_parent_child_transformations(T_world_joints):
    # Perfinger
    T_world_joints_root = T_world_joints[:, 0][:, None]
    T_world_joints_thumb = T_world_joints[:, 13:16]
    T_world_joints_index = T_world_joints[:, 1:4]
    T_world_joints_middle = T_world_joints[:, 4:7]
    T_world_joints_ring = T_world_joints[:, 10:13]
    T_world_joints_little = T_world_joints[:, 7:10]

    # Compute per finger transformations
    T_parent_child_thumb = compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_thumb) # [:, 3, 4, 4]
    T_parent_child_index = compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_index) # [:, 3, 4, 4]
    T_parent_child_middle = compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_middle) # [:, 3, 4, 4]
    T_parent_child_ring = compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_ring) # [:, 3, 4, 4]
    T_parent_child_little = compute_parent_child_transformations_per_finger(T_world_joints_root, T_world_joints_little) # [:, 3, 4, 4]

    results = {'thumb': T_parent_child_thumb,
               'index': T_parent_child_index,
               'middle': T_parent_child_middle,
               'ring': T_parent_child_ring,
               'little': T_parent_child_little} 
    return results
