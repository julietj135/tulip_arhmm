from preprocessing.dappy import read, preprocess, vis
import numpy as np
import preprocessing.ssumo.data.quaternion as qtn
from typing import List
from preprocessing.ssumo.data.dataset import *
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from preprocessing.augment import *
from keypointmoseq.jaxmoseq.models.keypoint_slds.alignment import align_egocentric
import jax

def get_root_rot(
    pose: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    forward_indices: Union[List, np.ndarray] = [0, 1, 2],
    target_direction: Union[List, np.ndarray] = [0, 1, 2],
):
    # Find forward root direction
    if len(forward_indices) == 3:
        forward = np.mean(np.cross(pose[:, forward_indices[1]] - pose[:, forward_indices[0]], pose[:, forward_indices[2]] - pose[:, forward_indices[0]]),axis=0)
    else:
        forward = pose[0, forward_indices[1], :] - pose[0, forward_indices[0], :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]
    
    # Root Rotation
    target = np.array([target_direction])
    target = np.array(target_direction)
    root_qtn = qtn.qbetween_np(forward, target)
    root_qtn = np.tile(root_qtn, (len(pose), 1))
    return root_qtn, qtn.quaternion_to_cont6d_np(root_qtn)

def inv_kin(
    pose: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    forward_indices: Union[List, np.ndarray] = [0, 1, 2],
    target_direction: Union[List, np.ndarray] = [0, 1, 2],
):
    root_quat, _ = get_root_rot(pose, kinematic_tree, offset, forward_indices, target_direction)
    local_quat = np.zeros(pose.shape[:-1] + (4,))

    root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]]) # try removing this because this sets og position as root but want to rotate it to the target vector
    
    local_quat[:, 0] = root_quat
    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]][None, ...].repeat(len(pose), axis=0)
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / np.linalg.norm(v, axis=-1)[..., None]
            rot_u_v = qtn.qbetween_np(u, v)
            R_loc = qtn.qmul_np(qtn.qinv_np(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qtn.qmul_np(R, R_loc)

    return local_quat

def convert_left_data(raw_pose, skeleton_config, forward_indices, target_direction):
    # apply inv kinematics
    local_qtn = inv_kin(
        raw_pose,
        skeleton_config["KINEMATIC_TREE"],
        np.array(skeleton_config["OFFSET"]),
        forward_indices=forward_indices,
        target_direction=target_direction
    )
    
    # Converting quaternions to 6d rotation representations
    return qtn.quaternion_to_cont6d_np(local_qtn)

def plot_points(keypoints, labels, image_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], color='blue')

    for i, label in enumerate(labels):
        ax.text(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], label, fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig("figures_videos/allsubs/{}".format(image_title))

def get_rotation_matrix(data,body,root_rot,forward_indices,num_frames):
    if body in "lr":
        matR = qtn.cont6d_to_matrix_np(root_rot)
    else:
        if len(forward_indices) == 3:
            curr_dir = np.mean(np.cross(data[:, forward_indices[1]] - data[:, forward_indices[0]], data[:, forward_indices[2]] - data[:, forward_indices[0]]),axis=0)
        else:
            curr_dir = np.mean(data[:, forward_indices[1]] - data[:, forward_indices[0]],axis=0)
        theta = np.arctan2(curr_dir[..., 1], curr_dir[..., 0])
        matR = np.array([[np.cos(-theta),-np.sin(-theta),0],
                         [np.sin(-theta),np.cos(-theta),0],
                         [0,0,1]])
        matR = np.tile(matR,(num_frames,1,1))
    return matR

def apply_rotation(data,body,root_rot,forward_indices,num_frames,z=False):
    """
    z = whether or not rotations are around the z axis or unrestricted (applies only to hand)
    """
    if len(forward_indices) == 3:
        curr_dir = np.mean(np.cross(data[:, forward_indices[1]] - data[:, forward_indices[0]], data[:, forward_indices[2]] - data[:, forward_indices[0]]),axis=0)
    else:
        curr_dir = np.mean(data[:, forward_indices[1]] - data[:, forward_indices[0]],axis=0)
    theta = np.arctan2(curr_dir[..., 1], curr_dir[..., 0])
    matR = np.array([[np.cos(-theta),-np.sin(-theta),0],
                        [np.sin(-theta),np.cos(-theta),0],
                        [0,0,1]])
    matR = np.tile(matR,(num_frames,1,1))
    if body in "lr" and not z:
        matR = qtn.cont6d_to_matrix_np(root_rot)
    
    return np.einsum('ijk,ikl->ijl', matR, data.transpose(0, 2, 1)).transpose(0, 2, 1) 

def kinematics(sub, skeleton_config_path, data, body, num_keypoints):
    if body in "lr":
        forward_indices = [0, 2, 17]
        target_direction = [1, 0, 0]
    else:
        subs=clean_gaits(sub, False)
        subs=subs[:8]
        forward_indices = [5, 6, 19]
        target_direction = [1, 0, 0]
        sub = subs[2]
            
    # Params
    num_frames = sub.shape[0]
    skeleton_config = read.config(skeleton_config_path)

    # Save raw pose in dataset if specified
    data["raw_pose"] = sub

    # Remove root
    root = sub[..., 0, :]
    duplicated_root = root[:, np.newaxis, :]
    duplicated_root = np.tile(duplicated_root, (1, num_keypoints, 1)) 
    data["raw_pose"] = data["raw_pose"] - duplicated_root
    root[..., [0, 1, 2]] = 0
    data["root"] = root
    
    # Rotate poses to face foward direction
    _, root_rot = get_root_rot(data["raw_pose"],
                            skeleton_config["KINEMATIC_TREE"],
                            np.array(skeleton_config["OFFSET"]),
                            forward_indices=forward_indices,
                            target_direction=target_direction)
        
    # Apply rotation twice
    data["rot_pose"] = apply_rotation(data["raw_pose"],body,root_rot,forward_indices,num_frames)

    # Get local 6D rotation representation 
    data["x6d"] = convert_left_data(data["rot_pose"],skeleton_config,forward_indices,target_direction)

    # Scale offsets by segment lengths
    data["offsets"] = get_segment_len(
        data["rot_pose"],
        skeleton_config["KINEMATIC_TREE"],
        np.array(skeleton_config["OFFSET"]),
    )

    reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
    
    data["target_pose"] = fwd_kin_cont6d(
                reshaped_x6d,
                skeleton_config["KINEMATIC_TREE"],
                data["offsets"],
                np.zeros((reshaped_x6d.shape[0], 3)), 
                True,
            )
    return data
        
def create_reconstructions(ids, body):
    data_keys = ["x6d", "root", "offsets", "raw_pose"]
    total_sub_error = []
    percerr_sub = []
    l2norm = []
    for id in ids:
        print("updating configuration for patient {}...".format(id), body)
        data = {}
        if body == "r":
            sub = pd.read_pickle(r'proj_directory/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Righthand_fingertapping/Filtered_medsav_rhand_Subject_{}.pkl'.format(id))
            skeleton_config_path = "preprocessing/configs/righthand_skeleton.yaml"
            num_keypoints = 21
            plot_points(sub[0,:,:], range(num_keypoints), "right_hand")
        elif body == "l":
            sub = pd.read_pickle(r'proj_directory/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Lefthand_fingertapping/Filtered_medsav_lhand_Subject_{}.pkl'.format(id))
            skeleton_config_path = "preprocessing/configs/lefthand_skeleton.yaml"
            num_keypoints = 21
            plot_points(sub[0,:,:], range(num_keypoints), "left_hand")
        elif body == "g":
            sub = pd.read_pickle(r'proj_directory/data/kpts_and_labels/all_sub_median_picker_3d_coordinates.pkl')[str(id)]
            skeleton_config_path = "preprocessing/configs/gait_skeleton.yaml"
            num_keypoints = 26
            plot_points(sub[0,:,:], range(num_keypoints), "gait")
        elif body == "lr":
            num_keypoints = 21
            skeleton_config_path = "preprocessing/configs/lefthand_skeleton.yaml"
            sub_right = pd.read_pickle(r'proj_directory/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Righthand_fingertapping/Filtered_medsav_rhand_Subject_{}.pkl'.format(id))
            sub_right_flip = sub_right.copy()
            sub_right_flip[...,0] = sub_right_flip[...,0]*-1
            sub_left = pd.read_pickle(r'proj_directory/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Lefthand_fingertapping/Filtered_medsav_lhand_Subject_{}.pkl'.format(id))
        
        if body == "lr":
            data_right = kinematics(sub_right_flip, skeleton_config_path, {}, body, num_keypoints)
            data_left = kinematics(sub_left, skeleton_config_path, {}, body, num_keypoints)
            num_frames = sub_left.shape[0]
        else:
            data = kinematics(sub, skeleton_config_path, data, body, num_keypoints)
            if body in "lr":
                data["Y_aligned"], v, h = align_egocentric(jax.device_put(data["raw_pose"]), [20], [4])
            elif body == "g":
                data["Y_aligned"], v, h = align_egocentric(jax.device_put(data["raw_pose"]), [0], [17])
            num_frames = data["raw_pose"].shape[0]
            
        connectivity = read.connectivity_config(skeleton_config_path)

        window = 100
        vis.pose.grid3D(
                    np.concatenate((data["raw_pose"], data["target_pose"], data["Y_aligned"]), axis=0), 
                    connectivity,
                    frames=[0, num_frames, num_frames*2],
                    centered=False,
                    labels=["Raw", "My Processing (chest)", "Keypoint Moseq (head)"],
                    title=None,
                    fps=80,
                    N_FRAMES=window,
                    VID_NAME="sub{}.2_chest_meanheading.mp4".format(id),
                    SAVE_ROOT="preprocessing/outputs/",
                )
            
        # FOR CALCULATING ERROR   
        percerr = 0
        totnorm = 0
        for frame in range(num_frames):
            for point in range(1, num_keypoints):
                norm = np.linalg.norm(data["rot_pose"][frame,point]-data["target_pose"][frame,point]) # unrot or target
                err = norm/np.linalg.norm(data["rot_pose"][frame,point])
                percerr+=err
                totnorm+=norm
        print("Error between raw pose and reconstructed for all frames and points: ", totnorm)
        total_sub_error.append(totnorm)
        percerr_sub.append(percerr/(num_frames*num_keypoints))
        l2norm.append(totnorm/num_frames)
        
    print("Average L2 norm across patients: ", sum(total_sub_error)/len(ids))
    print("Average percent error across patients: ", sum(percerr_sub)/len(ids))
    print("Average L2 norm per frame: ", sum(l2norm)/len(ids))

