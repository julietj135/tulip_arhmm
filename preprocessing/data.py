from preprocessing.augment import *
from preprocessing.kinematic_processing import *
from preprocessing.ssumo.data.quaternion import *

def get_raw_data(body, id, flipright=False):
    if body == "r":
        sub = pd.read_pickle(r'training/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Righthand_fingertapping/Filtered_medsav_rhand_Subject_{}.pkl'.format(id))
        skeleton_config = read.config("preprocessing/configs/righthand_skeleton.yaml")
        if flipright:
            sub[...,0] = sub[...,0]*-1
    elif body == "l" or body == "lr":
        sub = pd.read_pickle(r'training/data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Lefthand_fingertapping/Filtered_medsav_lhand_Subject_{}.pkl'.format(id))
        skeleton_config = read.config("preprocessing/configs/lefthand_skeleton.yaml")
    elif body == "g":
        sub = pd.read_pickle(r'training/data/kpts_and_labels/all_sub_median_picker_3d_coordinates.pkl')[str(id)]
        skeleton_config = read.config("preprocessing/configs/gait_skeleton.yaml")
    return sub, skeleton_config
            
def augment_data(sub, id, num_windows, body):
    if body in "lr":
        subs=augment_fingers(sub, num_windows, True)
        np.random.seed(id)
        indices = np.random.choice(len(subs),num_windows,replace=False)
        subs = [subs[ind] for ind in indices]
    else:
        subs=clean_gaits(sub, True)
        subs=subs[:num_windows]
    return subs

def standardize_data(sub, 
                     skeleton_config, 
                     body, 
                     forward_indices, 
                     target_direction, 
                     stand_offsets, 
                     get_offsets=False):
    num_frames, num_keypoints = sub.shape[0], sub.shape[1]         
    confidences = np.ones((num_frames, num_keypoints))
    
    root = sub[..., 0, :]
    duplicated_root = root[:, np.newaxis, :]
    duplicated_root = np.tile(duplicated_root, (1, num_keypoints, 1)) 
    sub = sub - duplicated_root
    rotated_pose = sub
    if body in "lr":
        # rotate for vertical alignment
        _, root_rot = get_root_rot(rotated_pose,
                            skeleton_config["KINEMATIC_TREE"],
                            np.array(skeleton_config["OFFSET"]),
                            forward_indices=[0,9],
                            target_direction=[0,0,1])
        
        rotated_pose = apply_rotation(rotated_pose,body,root_rot,forward_indices,num_frames,z=False)
        
    # # rotate for heading
    # _, root_rot = get_root_rot(sub,
    #                         skeleton_config["KINEMATIC_TREE"],
    #                         np.array(skeleton_config["OFFSET"]),
    #                         forward_indices=forward_indices,
    #                         target_direction=target_direction)
        
    rotated_pose = apply_rotation(rotated_pose,body,None,forward_indices,num_frames,z=True)

    rotation_coords = convert_left_data(rotated_pose,skeleton_config,forward_indices,target_direction)
    
    offsets = get_segment_len(
        rotated_pose, 
        skeleton_config["KINEMATIC_TREE"],
        np.array(skeleton_config["OFFSET"]),
    )
    
    if get_offsets:
        return offsets
    
    for i in range(1, np.array(skeleton_config["OFFSET"]).shape[0]):
        for j in range(offsets.shape[0]):
            offsets[j, i] = offsets[j,i]/np.linalg.norm(offsets[j,i])*np.linalg.norm(stand_offsets[0,i])
    
    # Forward kinematics
    reshaped_x6d = rotation_coords.reshape((-1,) + rotation_coords.shape[-2:])
    standardized_data = fwd_kin_cont6d(
                reshaped_x6d,
                skeleton_config["KINEMATIC_TREE"],
                offsets,
                np.zeros((reshaped_x6d.shape[0], 3)), 
                True)

    return sub, confidences, rotated_pose, rotation_coords, standardized_data
    
def get_augmented_data(ids, 
                       num_windows, 
                       body, 
                       forward_indices, 
                       target_direction):
    raw_sub_data = {}
    rot_sub_data = {}
    stand_sub_data = {}
    confidences = {}
    rot_pose = {}

    # get standardize offsets from patient 1
    sub, skeleton_config = get_raw_data(body, 1, False)
    stand_offsets = standardize_data(sub, skeleton_config, body, forward_indices, target_direction, None, True)
    
    if body == "lr":
        print("You should be using get_left_right_data, get_augmented_data is for one hand")
        return 
    
    for id in ids:
        sub, skeleton_config = get_raw_data(body, id, False)
        subs = augment_data(sub, id, num_windows, body)
        print("subject {} has an augmentation of {} segments".format(id,len(subs)))
        for l in range(len(subs)):
            name = "sub{}.{}".format(id,l)
            sub_seg = subs[l]
            raw_sub_data[name], confidences[name], rot_pose[name], rot_sub_data[name], stand_sub_data[name] = standardize_data(sub_seg, skeleton_config, body, forward_indices, target_direction, stand_offsets, False)        
    
    return {"raw" : raw_sub_data, "conf" : confidences, "rot" : rot_pose, "x6d" : rot_sub_data, "stand" : stand_sub_data}


def get_subject_data(ids, num_windows, body, forward_indices, target_direction):
    raw_sub_data = {}
    rot_sub_data = {}
    stand_sub_data = {}
    confidences = {}
    rot_pose = {}
    
    # get standardize offsets from patient 1
    sub, skeleton_config = get_raw_data(body, 1, False)
    stand_offsets = standardize_data(sub, skeleton_config, body, forward_indices, target_direction, None, True)
    
    for id in ids:
        print("updating configuration for patient {}...".format(id))
        sub, skeleton_config = get_raw_data(body, id, False)
        name = "sub{}".format(id)
        raw_sub_data[name], confidences[name], rot_pose[name], rot_sub_data[name], stand_sub_data[name] = standardize_data(sub, skeleton_config, body, forward_indices, target_direction, stand_offsets, False)        
    
    return {"raw" : raw_sub_data, "conf" : confidences, "rot" : rot_pose, "x6d" : rot_sub_data, "stand" : stand_sub_data}

def get_left_right_data(ids, num_windows, body, forward_indices, target_direction):
    raw_sub_data = {}
    rot_sub_data = {}
    stand_sub_data = {}
    confidences = {}
    rot_pose = {}
    
    # get standardize offsets from patient 1
    sub, skeleton_config = get_raw_data(body, 1, False)
    stand_offsets = standardize_data(sub, skeleton_config, body, forward_indices, target_direction, None, True)
    
    if body != "lr":
        print("You should not be using get_left_right_data")
        return 
    
    for id in ids:
        for bod in "lr":
            sub, skeleton_config = get_raw_data(bod, id, flipright=True)
            subs = augment_data(sub, id, num_windows, bod)
            print("{} subject {} has an augmentation of {} segments".format(bod, id,len(subs)))
            for l in range(len(subs)):
                name = "{}sub{}.{}".format(bod,id,l)
                sub_seg = subs[l]
                raw_sub_data[name], confidences[name], rot_pose[name], rot_sub_data[name], stand_sub_data[name] = standardize_data(sub_seg, skeleton_config, bod, forward_indices, target_direction, stand_offsets, False)        

    return {"raw" : raw_sub_data, "conf" : confidences, "rot" : rot_pose, "x6d" : rot_sub_data, "stand" : stand_sub_data}
   