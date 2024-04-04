import os
import cv2
import json
import torch
import random
import numpy as np
import pandas as pd
import imageio as io
# from skimage import io
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops_table
from pyometiff import OMETIFFReader


def calculate_normalization(channel_path, quantile):
    """Calculates the normalization value for a given channel
    Args:
        channel_path (str): path to channel
        quantile (float): quantile to use for normalization
    Returns:
        normalization_value (float): normalization value
    """
    mplex_img = io.imread(channel_path)
    mplex_img = mplex_img.astype(np.float32)
    foreground = mplex_img[mplex_img > 0]
    normalization_value = np.quantile(foreground, quantile)
    chan = os.path.basename(channel_path).split(".")[0]
    return chan, normalization_value


def prepare_normalization_dict(
        fov_paths, output_dir, quantile=0.999, include_channels=[], n_subset=10, n_jobs=1,
        output_name="normalization_dict.json"
    ):
    """Prepares the normalization dict for a list of fovs
    Args:
        fov_paths (list): list of paths to fovs
        output_dir (str): path to output directory
        quantile (float): quantile to use for normalization
        exclude_channels (list): list of channels to exclude
        n_subset (int): number of fovs to use for normalization
        n_jobs (int): number of jobs to use for joblib multiprocessing
        output_name (str): name of output file
    Returns:
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
    """
    normalization_dict = {}
    if n_subset is not None:
        random.shuffle(fov_paths)
        fov_paths = fov_paths[:n_subset]
    print("Iterate over fovs...")
    for fov_path in tqdm(fov_paths):
        channels = os.listdir(fov_path)
        if include_channels:
            channels = [
                channel for channel in channels if channel.split(".")[0] in include_channels
            ]
        channel_paths = [os.path.join(fov_path, channel) for channel in channels]
        if n_jobs > 1:
            normalization_values = Parallel(n_jobs=n_jobs)(
            delayed(calculate_normalization)(channel_path, quantile)
            for channel_path in channel_paths
            )
        else:
            normalization_values = [
                calculate_normalization(channel_path, quantile)
                for channel_path in channel_paths
            ]
        for channel, normalization_value in normalization_values:
            if channel not in normalization_dict:
                normalization_dict[channel] = []
            normalization_dict[channel].append(normalization_value)
    if n_jobs > 1:
        get_reusable_executor().shutdown(wait=True)
    for channel in normalization_dict.keys():
        normalization_dict[channel] = np.mean(normalization_dict[channel])
    # save normalization dict
    with open(os.path.join(output_dir, output_name), 'w') as f:
        json.dump(normalization_dict, f)
    return normalization_dict


def prepare_input_data(mplex_img, instance_mask):
    """Prepares the input data for the segmentation model
    Args:
        mplex_img (np.array): multiplex image
        instance_mask (np.array): instance mask
    Returns:
        input_data (np.array): input data for segmentation model
    """
    mplex_img = mplex_img.astype(np.float32)
    edge = find_boundaries(instance_mask, mode="inner").astype(np.uint8)
    binary_mask = np.logical_and(edge == 0, instance_mask > 0).astype(np.float32)
    input_data = np.stack([mplex_img, binary_mask], axis=0)[np.newaxis,...] # bhwc
    return input_data


def segment_mean(instance_mask, prediction):
    """Calculates the mean prediction per instance
    Args:
        instance_mask (np.array): instance mask
        prediction (np.array): prediction
    Returns:
        uniques (np.array): unique instance ids
        mean_per_cell (np.array): mean prediction per instance
    """
    props_df = regionprops_table(
        label_image=instance_mask, intensity_image=prediction,
        properties=['label' ,'intensity_mean']
    )
    return props_df


def test_time_aug(
        input_data, channel, app, normalization_dict, rotate=True, flip=True, batch_size=4
    ):
    """Performs test time augmentation
    Args:
        input_data (np.array): input data for segmentation model, mplex_img and binary mask
        channel (str): channel name
        app (tf.keras.Model): segmentation model
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        rotate (bool): whether to rotate
        flip (bool): whether to flip
        batch_size (int): batch size
    Returns:
        seg_map (np.array): predicted segmentation map
    """
    forward_augmentations = []
    backward_augmentations = []
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data)
    if rotate:
        for k in [0,1,2,3]:
            forward_augmentations.append(lambda x: torch.rot90(x, k=k, dims=[2,3]))
            backward_augmentations.append(lambda x: torch.rot90(x, k=-k, dims=[2,3]))
    if flip:
        forward_augmentations += [
            lambda x: torch.flip(x, [2]),
            lambda x: torch.flip(x, [3])
        ]
        backward_augmentations += [
            lambda x: torch.flip(x, [2]),
            lambda x: torch.flip(x, [3])
        ]
    input_batch = []
    for forw_aug in forward_augmentations:
        input_data_tmp = forw_aug(input_data).numpy() # bhwc
        input_batch.append(np.concatenate(input_data_tmp))
    input_batch = np.stack(input_batch, 0)
    seg_map = app.predict_segmentation(
        input_batch,
        preprocess_kwargs={
            "normalize": True,
            "marker": channel,
            "normalization_dict": normalization_dict},
        )
    seg_map = torch.from_numpy(seg_map)
    tmp = []
    for backw_aug, seg_map_tmp in zip(backward_augmentations, seg_map):
        seg_map_tmp = backw_aug(seg_map_tmp[np.newaxis,...])
        seg_map_tmp = np.squeeze(seg_map_tmp)
        tmp.append(seg_map_tmp)
    seg_map = np.stack(tmp, 0)
    seg_map = np.mean(seg_map, axis = 0)
    return seg_map


def predict_fovs(
        nimbus, fov_paths, normalization_dict, segmentation_naming_convention, output_dir,
        suffix, include_channels=[], save_predictions=True, half_resolution=False, batch_size=4,
        test_time_augmentation=True
    ):
    """Predicts the segmentation map for each mplex image in each fov
    Args:
        nimbus (Nimbus): nimbus object
        fov_paths (list): list of fov paths
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        segmentation_naming_convention (function): function to get instance mask path from fov path
        output_dir (str): path to output dir
        suffix (str): suffix of mplex images
        include_channels (list): list of channels to include
        save_predictions (bool): whether to save predictions
        half_resolution (bool): whether to use half resolution
        batch_size (int): batch size
        test_time_augmentation (bool): whether to use test time augmentation
    Returns:
        cell_table (pd.DataFrame): cell table with predicted confidence scores per fov and cell
    """
    fov_dict_list = []
    for fov_path in fov_paths:
        print(f"Predicting {fov_path}...")
        out_fov_path = os.path.join(
            os.path.normpath(output_dir), os.path.basename(fov_path)
        )
        df_fov = pd.DataFrame()
        instance_path = segmentation_naming_convention(fov_path)
        instance_mask = np.squeeze(io.imread(instance_path))
        for channel in tqdm(os.listdir(fov_path)):
            channel_path = os.path.join(fov_path, channel)
            channel_ = channel.split(".")[0]
            if not channel.endswith(suffix) or (
                include_channels != [] and channel_ not in include_channels
                ):
                continue
            mplex_img = np.squeeze(io.imread(channel_path))
            input_data = prepare_input_data(mplex_img, instance_mask)
            if half_resolution:
                scale = 0.5
                input_data = np.squeeze(input_data)
                _, h,w = input_data.shape
                img = cv2.resize(input_data[0], [int(h*scale), int(w*scale)])
                binary_mask = cv2.resize(
                    input_data[1], [int(h*scale), int(w*scale)], interpolation=0
                )
                input_data = np.stack([img, binary_mask], axis=0)[np.newaxis,...]
            if test_time_augmentation:
                prediction = test_time_aug(
                    input_data, channel, nimbus, normalization_dict, batch_size=batch_size
                )
            else:
                prediction = nimbus.predict_segmentation(
                    input_data,
                    preprocess_kwargs={
                        "normalize": True, "marker": channel,
                        "normalization_dict": normalization_dict
                    },
                )
            prediction = np.squeeze(prediction)
            if half_resolution:
                prediction = cv2.resize(prediction, (h, w))
            df = pd.DataFrame(segment_mean(instance_mask, prediction))
            if df_fov.empty:
                df_fov["label"] = df["label"]
                df_fov["fov"] = os.path.basename(fov_path)
            df_fov[channel.split(".")[0]] = df["intensity_mean"]
            if save_predictions:
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = (prediction*255.0).astype(np.uint8)
                io.imwrite(
                    os.path.join(out_fov_path, channel), pred_int, photometric="minisblack",
                    # compress=0, 
                )
        fov_dict_list.append(df_fov)
    cell_table = pd.concat(fov_dict_list, ignore_index=True)
    return cell_table


def nimbus_preprocess(image, **kwargs):
    """Preprocess input data for Nimbus model.
    Args:
        image: array to be processed
    Returns:
        np.array: processed image array
    """
    output = np.copy(image.astype(np.float32))
    if len(image.shape) != 4:
        raise ValueError("Image data must be 4D, got image of shape {}".format(image.shape))

    normalize = kwargs.get('normalize', True)
    if normalize:
        marker = kwargs.get('marker', None)
        normalization_dict = kwargs.get('normalization_dict', {})
        if marker in normalization_dict.keys():
            norm_factor = normalization_dict[marker]
        else:
            print("Norm_factor not found for marker {}, calculating directly from the image. \
            ".format(marker))
            norm_factor = np.quantile(output[..., 0], 0.999)
        # normalize only marker channel in chan 0 not binary mask in chan 1
        output[..., 0] /= norm_factor
        output = output.clip(0, 1)
    return output


def calculate_normalization_ome(ome_path, quantile, include_channels):
    """Calculates the normalization values for a given ome file
    Args:
        ome_path (str): path to ome file
        quantile (float): quantile to use for normalization
        include_channels (list): list of channels to include
    Returns:
        normalization_values (dict): dict with channel names as keys and norm factors  as values
    """
    reader = OMETIFFReader(fpath=ome_path)
    img_array, metadata, _ = reader.read()
    channel_names = list(metadata["Channels"].keys())
    if not include_channels:
        include_channels = channel_names
    # check if include_channels are included in ome file metadata
    for channel in include_channels:
        if channel not in channel_names:
            raise ValueError(f"Channel {channel} not found in ome file metadata.")
    normalization_values = {}
    for channel in include_channels:
        idx = channel_names.index(channel)
        mplex_img = img_array[idx]
        mplex_img = mplex_img.astype(np.float32)
        foreground = mplex_img[mplex_img > 0]
        normalization_values[channel] = np.quantile(foreground, quantile)
    return normalization_values


def prepare_normalization_dict_ome(
        fov_paths, output_dir, quantile=0.999, include_channels=[], n_subset=10, n_jobs=1,
        output_name="normalization_dict.json"
    ):
    """Prepares the normalization dict for a list of ome.tif fovs
    Args:
        fov_paths (list): list of paths to fovs
        output_dir (str): path to output directory
        quantile (float): quantile to use for normalization
        exclude_channels (list): list of channels to exclude
        n_subset (int): number of fovs to use for normalization
        n_jobs (int): number of jobs to use for joblib multiprocessing
        output_name (str): name of output file
    Returns:
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
    """
    normalization_dict = {}
    if n_subset is not None:
        random.shuffle(fov_paths)
        fov_paths = fov_paths[:n_subset]
    print("Iterate over fovs...")
    if n_jobs > 1:
        normalization_values = Parallel(n_jobs=n_jobs)(
            delayed(calculate_normalization_ome)(ome_path, quantile, include_channels)
            for ome_path in fov_paths
        )
    else:
        normalization_values = [
            calculate_normalization_ome(ome_path, quantile, include_channels)
            for ome_path in fov_paths
        ]
    for norm_dict in normalization_values:
        for channel, normalization_value in norm_dict.items():
            if channel not in normalization_dict:
                normalization_dict[channel] = []
            normalization_dict[channel].append(normalization_value)
    if n_jobs > 1:
        get_reusable_executor().shutdown(wait=True)
    for channel in normalization_dict.keys():
        normalization_dict[channel] = np.mean(normalization_dict[channel])
    # save normalization dict
    with open(os.path.join(output_dir, output_name), 'w') as f:
        json.dump(normalization_dict, f)
    return normalization_dict


def predict_ome_fovs(
        nimbus, fov_paths, normalization_dict, segmentation_naming_convention, output_dir,
        suffix, include_channels=[], save_predictions=True, half_resolution=False, batch_size=4,
        test_time_augmentation=True
    ):
    """Predicts the segmentation map for each mplex channel in each ome.tif fov
    Args:
        nimbus (Nimbus): nimbus object
        fov_paths (list): list of fov paths
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        segmentation_naming_convention (function): function to get instance mask path from fov path
        output_dir (str): path to output dir
        suffix (str): suffix of mplex images
        include_channels (list): list of channels to include
        save_predictions (bool): whether to save predictions
        half_resolution (bool): whether to use half resolution
        batch_size (int): batch size
        test_time_augmentation (bool): whether to use test time augmentation
    Returns:
        cell_table (pd.DataFrame): cell table with predicted confidence scores per fov and cell
    """
    fov_dict_list = []
    for fov_path in fov_paths:
        print(f"Predicting {fov_path}...")
        out_fov_path = os.path.join(
            os.path.normpath(output_dir), os.path.basename(fov_path).split(".")[0]
        )
        df_fov = pd.DataFrame()
        reader = OMETIFFReader(fpath=fov_path)
        img_array, metadata, _ = reader.read()
        channel_names = list(metadata["Channels"].keys())
        instance_path = segmentation_naming_convention(fov_path)
        instance_mask = np.squeeze(io.imread(instance_path))
        if not include_channels:
            include_channels = channel_names
        for channel in tqdm(include_channels):
            idx = channel_names.index(channel)
            mplex_img = np.squeeze(img_array[idx])
            input_data = prepare_input_data(mplex_img, instance_mask)
            if half_resolution:
                scale = 0.5
                input_data = np.squeeze(input_data)
                _, h,w = input_data.shape
                img = cv2.resize(input_data[0], [int(h*scale), int(w*scale)])
                binary_mask = cv2.resize(
                    input_data[1], [int(h*scale), int(w*scale)], interpolation=0
                )
                input_data = np.stack([img, binary_mask], axis=0)[np.newaxis,...]
            if test_time_augmentation:
                prediction = test_time_aug(
                    input_data, channel, nimbus, normalization_dict, batch_size=batch_size
                )

            else:
                prediction = nimbus.predict_segmentation(
                    input_data,
                    preprocess_kwargs={
                        "normalize": True, "marker": channel,
                        "normalization_dict": normalization_dict
                    },
                )
            prediction = np.squeeze(prediction)
            if half_resolution:
                prediction = cv2.resize(prediction, (h, w))
            df = pd.DataFrame(segment_mean(instance_mask, prediction))
            if df_fov.empty:
                df_fov["label"] = df["label"]
                df_fov["fov"] = os.path.basename(fov_path)
            df_fov[channel] = df["intensity_mean"]
            if save_predictions:
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = (prediction*255.0).astype(np.uint8)
                io.imwrite(
                    os.path.join(out_fov_path, channel+".tiff"), pred_int, photometric="minisblack",
                    # compress=0, 
                )
        fov_dict_list.append(df_fov)
    cell_table = pd.concat(fov_dict_list, ignore_index=True)
    return cell_table
