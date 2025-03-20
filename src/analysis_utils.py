import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation
import torch

# torch.manual_seed(1234)
# np.random.seed(12345)

def quaternion_to_6d_batch(quats):
    """
    CVPR: On the continuity of rotation representations in neural networks'19
    """
    quats = np.asarray(quats)
    # Create a batch Rotation object from an (N,4) array
    rot_batch = Rotation.from_quat(quats)  # shape (N,)
    
    # Extract the 3x3 matrices, shape (N, 3, 3)
    R = rot_batch.as_matrix()
    
    # First two columns => shape (N, 3) each
    r1 = R[:, :, 0]  # shape (N, 3)
    r2 = R[:, :, 1]  # shape (N, 3)
    
    # Concatenate along axis=1 => shape (N, 6)
    r6d = np.concatenate([r1, r2], axis=1)
    return r6d


# Function to extract and convert vectors to tensors
def prepare_vectors(df):
    force_cols = ['force_x', 'force_y', 'force_z', 
                  "torque_x", "torque_y", "torque_z",]
    output_cols = ['vel_x', 'vel_y', 'vel_z', 'angvel_x', 'angvel_y', 'angvel_z']
    feature_cols = [col for col in df.columns if col not in (force_cols + output_cols)]
    
    dist_vec = df[feature_cols].values
    force_vec = df[force_cols].values
    output_vec = df[output_cols].values
    return dist_vec, force_vec, output_vec

# convert numpy arrays to PyTorch tensors
def convert_to_tensors(dist_vec, ft_vec, output_vec):
    dist_tensor = torch.tensor(dist_vec, dtype=torch.float32)  
    force_tensor = torch.tensor(ft_vec, dtype=torch.float32) 
    velocity_tensor = torch.tensor(output_vec, dtype=torch.float32)
    return dist_tensor, force_tensor, velocity_tensor

# Function to shuffle and split the data
def shuffle_and_split(df, dist_tensor, force_tensor, velocity_tensor, split_frac=.80):
    indices = torch.randperm(dist_tensor.size(0))
    shuffled_df = df.iloc[indices.tolist()].reset_index(drop=True)
    dist_tensor = dist_tensor[indices]
    force_tensor = force_tensor[indices]
    velocity_tensor = velocity_tensor[indices]

    split_idx = int(split_frac * dist_tensor.size(0))
    
    # Split the tensors
    train_dist_tensor = dist_tensor[:split_idx]
    val_dist_tensor = dist_tensor[split_idx:]
    train_force_tensor = force_tensor[:split_idx]
    val_force_tensor = force_tensor[split_idx:]
    train_velocity_tensor = velocity_tensor[:split_idx]
    val_velocity_tensor = velocity_tensor[split_idx:]

    return (train_dist_tensor, val_dist_tensor, 
            train_force_tensor, val_force_tensor, 
            train_velocity_tensor, val_velocity_tensor)