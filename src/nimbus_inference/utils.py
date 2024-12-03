import os
import cv2
import json
import torch
import random
import numpy as np
import pandas as pd
import imageio as io
from copy import copy
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops_table
from pyometiff import OMETIFFReader
from pyometiff.omexml import OMEXML
from alpineer import io_utils, misc_utils
from typing import Callable
import tifffile
import zarr
import sys, os
import logging
import os, sys
import lmdb


class HidePrints:
    """Context manager to hide prints"""
    def __enter__(self):
        """Hide prints"""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Show prints"""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class LazyOMETIFFReader(OMETIFFReader):
    """Lazy OMETIFFReader class that reads channels only when needed

    Args:
        fpath (str): path to ome.tif file
    """
    def __init__(self, fpath: str):
        super().__init__(fpath)
        self.metadata = self.get_metadata()
        self.channels = self.get_channel_names()
        self.shape = self.get_shape()
    
    def get_metadata(self):
        """Get the metadata of the OME-TIFF file

        Returns:
            dict: metadata of the OME-TIFF file
        """
        with tifffile.TiffFile(str(self.fpath)) as tif:
            if tif.is_ome:
                omexml_string = tif.ome_metadata
                with HidePrints():
                    metadata = self.parse_metadata(omexml_string)
                return metadata
            else:
                raise ValueError("File is not an OME-TIFF file.")

    def get_channel_names(self):
        """Get the channel names of the OME-TIFF file

        Returns:
            list: list of channel names
        """
        if hasattr(self, "metadata"):
            return list(self.metadata["Channels"].keys())
        else:
            return []
    
    def get_shape(self):
        """Get the shape of the OME-TIFF file array data

        Returns:
            tuple: shape of the array data
        """
        with tifffile.imread(str(self.fpath), aszarr=True) as store:
            z = zarr.open(store, mode='r')
            if hasattr(z, "shape"):
                zz = z
            else:
                zz = z[0]
            shape = zz.shape
        return shape

    def get_channel(self, channel_name: str):
        """Get an individual channel from the OME-TIFF file by name

        Args:
            channel_name (str): name of the channel
        Returns:
            np.array: channel image
        """
        idx = self.channels.index(channel_name)
        with tifffile.imread(str(self.fpath), aszarr=True) as store:
            z = zarr.open(store, mode='r')
            # correct DimOrder, often DimOrder is TZCYX, but image is stored as CYX,
            # thus we remove the trailing dimensions
            if hasattr(z, "shape"):
                zz = z
            else:
                zz = z[0]
            dim_order = self.metadata["DimOrder"]
            if len(dim_order) > len(zz.shape):
                dim_order = dim_order.replace("T", "")
            if len(dim_order) > len(zz.shape):
                dim_order = dim_order.replace("Z", "")
            channel_idx = dim_order.find("C")
            slice_string = "zz[" + ":," * channel_idx + str(idx) + "]"
            channel = eval(slice_string)
        return channel


def _handle_qupath_segmentation_map(instance_mask: np.array):
    """Handle QuPath segmentation maps stored as 24-bit RGB images. The RGB channels are combined
    into a single channel via the following formula: label = RED*256**2 + GREEN * 256 + BLUE

    Args:
        instance_mask (np.array): instance mask
    Returns:
        np.array: instance mask
    """
    # add warning
    logging.warning("QuPath RGB segmentation map detected. Converting to instance map by")
    logging.warning("combining the RGB channels into a single channel via the following formula:")
    logging.warning("label = RED*256**2 + GREEN * 256 + BLUE")
    # move channel axis to last if not already
    if instance_mask.shape.index(3) == 0:
        instance_mask = np.moveaxis(instance_mask, 0, -1)
    instance_mask_handled = instance_mask[..., 0] * 256**2 + instance_mask[..., 1] * 256 \
        + instance_mask[..., 2]
    instance_mask_handled = instance_mask_handled.round(0).astype(np.uint64)
    return instance_mask_handled


class MultiplexDataset():
    """Multiplex dataset class that gives a common interface for data loading of multiplex
    datasets stored as individual channel images in folders or as multi-channel tiffs.

    Args:
        fov_paths (list): list of paths to fovs
        segmentation_naming_convention (function): function to get instance mask path from fov
            path
        suffix (str): suffix of channel images
        silent (bool): whether to print messages
        groundtruth_df (pd.DataFrame): groundtruth dataframe with columns fov, cell_id, channel and
            activity (0: negative, 1: positive, 2: ambiguous)
        magnification (int): magnification factor of the images (default: 20)
        validation_fovs (list): list of fovs to use for validation
        output_dir (str): path to output directory
    """
    def __init__(
            self, fov_paths: list, segmentation_naming_convention: Callable = None,
            include_channels: list = [], suffix: str = ".tiff", silent: bool = False,
            groundtruth_df: pd.DataFrame = None, magnification: int = 20,
            validation_fovs: list = [], output_dir: str = ""
        ):
        self.fov_paths = fov_paths
        self.segmentation_naming_convention = segmentation_naming_convention
        self.suffix = suffix
        if self.suffix[0] != ".": 
            self.suffix = "." + self.suffix
        self.silent = silent
        self.include_channels = include_channels
        self.multi_channel = self.is_multi_channel_tiff(fov_paths[0])
        self.channels = self.get_channels()
        self.check_inputs()
        self.fovs = self.get_fovs()
        self.channels = self.filter_channels(self.channels)
        self.groundtruth_df = groundtruth_df
        self.magnification = magnification
        self.output_dir = output_dir
        
        if validation_fovs and groundtruth_df is not None:
            self.validation_fovs = validation_fovs
            self.training_fovs = [fov for fov in self.fovs if fov not in self.validation_fovs]
        elif groundtruth_df is not None:
            num_validation_fovs = len(self.fovs)//10 if len(self.fovs) > 10 else 1
            self.validation_fovs = self.fovs[-num_validation_fovs:]
            self.training_fovs = self.fovs[:-num_validation_fovs]

    def filter_channels(self, channels):
        """Filter channels based on include_channels

        Args:
            channels (list): list of channel names
        Returns:
            list: filtered list of channel names
        """
        if self.include_channels:
            return [channel for channel in channels if channel in self.include_channels]
        return channels
    
    def get_groundtruth(self, fov: str, channel: str):	
        """Get the groundtruth for a fov / channel combination

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: groundtruth activity mask (0: negative, 1: positive, 2: ambiguous)
        """
        if self.groundtruth_df is None:
            raise ValueError("No groundtruth dataframe provided.")
        subset_df = self.groundtruth_df[
            (self.groundtruth_df["fov"] == fov) & (self.groundtruth_df["channel"] == channel)
        ]
        positive_cells = subset_df[subset_df["activity"] == 1].cell_id.values
        ambiguous_cells = subset_df[subset_df["activity"] == 2].cell_id.values
        instance_mask = self.get_segmentation(fov)
        groundtruth = np.zeros_like(instance_mask)
        # get all positions of positive cells in instance mask without a for loop
        positive_positions = np.where(np.isin(instance_mask, positive_cells))
        ambiguous_positions = np.where(np.isin(instance_mask, ambiguous_cells))
        groundtruth[positive_positions] = 1
        groundtruth[ambiguous_positions] = 2
        return groundtruth[np.newaxis,...] # 1, h, w

    def check_inputs(self):
        """Check inputs for Nimbus model"""
        # check if all paths in fov_paths exists
        if not isinstance(self.fov_paths, (list, tuple)):
            self.fov_paths = [self.fov_paths]
        io_utils.validate_paths(self.fov_paths)
        if isinstance(self.include_channels, str):
            self.include_channels = [self.include_channels]
        misc_utils.verify_in_list(
            include_channels=self.include_channels, dataset_channels=self.channels
        )
        if not self.silent:
            print("All inputs are valid")

    def __len__(self):
        """Return the number of fovs in the dataset"""
        return len(self.fov_paths)
    
    def is_multi_channel_tiff(self, fov_path: str):
        """Check if fov is a multi-channel tiff

        Args:
            fov_path (str): path to fov
        Returns:
            bool: whether fov is multi-channel
        """
        multi_channel = False
        if fov_path.lower().endswith(("ome.tif", "ome.tiff")):
            self.img_reader = LazyOMETIFFReader(fov_path)
            if len(self.img_reader.shape) > 2:
                multi_channel = True
        return multi_channel
    
    def get_channels(self):
        """Get the channel names for the dataset"""
        if self.multi_channel:
            return self.img_reader.channels
        else:
            channels = [
                channel.replace(self.suffix, "") for channel in os.listdir(self.fov_paths[0]) \
                    if channel.endswith(self.suffix)
            ]
            return channels
    
    def get_fovs(self):
        """Get the fovs in the dataset"""
        return [os.path.basename(fov).replace(self.suffix, "") for fov in self.fov_paths]
    
    def get_channel(self, fov: str, channel: str):
        """Get a channel from a fov

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: channel image
        """
        if self.multi_channel:
            return self.get_channel_stack(fov, channel)
        else:
            return self.get_channel_single(fov, channel)
        
    def get_channel_normalized(self, fov: str, channel: str):
        """Get a channel from a fov and normalize it

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: channel image
        """
        if not hasattr(self, "normalization_dict"):
            print("No normalization dict found. Preparing normalization dict...")
            self.prepare_normalization_dict()
        mplex_img = self.get_channel(fov, channel)
        mplex_img = mplex_img.astype(np.float32)
        if channel in self.normalization_dict.keys():
            norm_factor = self.normalization_dict[channel]
        else:
            norm_factor = np.quantile(mplex_img, 0.999)
        mplex_img /= norm_factor
        mplex_img = mplex_img.clip(0, 1)
        return mplex_img

    def get_channel_single(self, fov: str, channel: str):
        """Get a channel from a fov stored as a folder with individual channel images

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: channel image
        """
        idx = self.fovs.index(fov)
        fov_path = self.fov_paths[idx]
        channel_path = os.path.join(fov_path, channel + self.suffix)
        mplex_img = np.squeeze(io.imread(channel_path))
        return mplex_img

    def get_channel_stack(self, fov: str, channel: str):
        """Get a channel from a multi-channel tiff

        Args:
            fov (str): name of a fov
            channel (str): channel name
            data_format (str): data format
        Returns:
            np.array: channel image
        """
        idx = self.fovs.index(fov)
        fov_path = self.fov_paths[idx]
        self.img_reader = LazyOMETIFFReader(fov_path)
        return np.squeeze(self.img_reader.get_channel(channel))
    
    def get_segmentation(self, fov: str):
        """Get the instance mask for a fov

        Args:
            fov (str): name of a fov
        Returns:
            np.array: instance mask
        """
        idx = self.fovs.index(fov)
        fov_path = self.fov_paths[idx]
        instance_path = self.segmentation_naming_convention(fov_path)
        if isinstance(instance_path, str):
            instance_mask = io.imread(instance_path)
        else:
            instance_mask = instance_path
        instance_mask = np.squeeze(instance_mask)
        if len(instance_mask.shape) == 3:
            instance_mask = _handle_qupath_segmentation_map(instance_mask)
        instance_mask = instance_mask.astype(np.uint32)
        return instance_mask

    def prepare_normalization_dict(
        self, quantile=0.999, clip_values=(0, 2), n_subset=10, multiprocessing=False,
        overwrite=False,
    ):
        """Load or prepare and save normalization dictionary for Nimbus model.

        Args:
            quantile (float): Quantile to use for normalization.
            clip_values (list): Values to clip images to after normalization.
            n_subset (int): Number of fovs to use for normalization.
            multiprocessing (bool): Whether to use multiprocessing.
            overwrite (bool): Whether to overwrite existing normalization dict.
        Returns:
            dict: Dictionary of normalization factors.
        """
        self.clip_values = tuple(clip_values)
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")
        if os.path.exists(self.normalization_dict_path) and not overwrite:
            self.normalization_dict = json.load(open(self.normalization_dict_path))
            self.normalization_dict = {k: float(v) for k, v in self.normalization_dict.items()}
        else:
            n_jobs = os.cpu_count() if multiprocessing else 1
            self.normalization_dict = prepare_normalization_dict(
                self, self.output_dir, quantile, n_subset,
                n_jobs
            )

def prepare_input_data(mplex_img, instance_mask):
    """Prepares the input data for the segmentation model

    Args:
        mplex_img (np.array): multiplex image
        instance_mask (np.array): instance mask
    Returns:
        np.array: input data for segmentation model
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
        np.array: unique instance ids
        np.array: mean prediction per instance
    """
    props_df = regionprops_table(
        label_image=instance_mask.astype(np.int32), intensity_image=prediction,
        properties=['label' , 'centroid', 'intensity_mean']
    )
    return props_df


def test_time_aug(
        input_data, channel, app, normalization_dict, rotate=True, flip=True, batch_size=4,
        clip_values=(0, 2)
    ):
    """Performs test time augmentation

    Args:
        input_data (np.array): input data for segmentation model, mplex_img and binary mask
        channel (str): channel name
        app (Nimbus): segmentation model
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        rotate (bool): whether to rotate
        flip (bool): whether to flip
        batch_size (int): batch size
    Returns:
        np.array: predicted segmentation map
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
    output = []
    for forw_aug, backw_aug in zip(forward_augmentations, backward_augmentations):
        input_data_aug = forw_aug(input_data).numpy() # bhwc
        seg_map = app.predict_segmentation(input_data_aug)
        if not isinstance(seg_map, torch.Tensor):
            seg_map = torch.from_numpy(seg_map)
        seg_map = backw_aug(seg_map)
        seg_map = np.squeeze(seg_map)
        output.append(seg_map)
    seg_map = np.stack(output, 0)
    seg_map = np.mean(seg_map, axis = 0)
    return seg_map


def predict_fovs(
        nimbus, dataset: MultiplexDataset, output_dir: str, suffix: str=".tiff",
        save_predictions: bool=True, batch_size: int=4, test_time_augmentation: bool=True
    ):
    """Predicts the segmentation map for each mplex image in each fov

    Args:
        nimbus (Nimbus): nimbus object
        dataset (MultiplexDataset): dataset object
        output_dir (str): path to output dir
        suffix (str): suffix of mplex images
        save_predictions (bool): whether to save predictions
        batch_size (int): batch size
        test_time_augmentation (bool): whether to use test time augmentation
    Returns:
        pd.DataFrame: cell table with predicted confidence scores per fov and cell
    """
    fov_dict_list = []
    for fov_path, fov in zip(dataset.fov_paths, dataset.fovs):
        print(f"Predicting {fov_path}...")
        out_fov_path = os.path.join(
            os.path.normpath(output_dir), os.path.basename(fov_path).replace(suffix, "")
        )
        df_fov = pd.DataFrame()
        instance_mask = dataset.get_segmentation(fov)
        for channel_name in tqdm(dataset.channels):
            mplex_img = dataset.get_channel_normalized(fov, channel_name)
            input_data = prepare_input_data(mplex_img, instance_mask)
            if dataset.magnification != nimbus.model_magnification:
                scale = nimbus.model_magnification / dataset.magnification
                input_data = np.squeeze(input_data)
                _, h,w = input_data.shape
                img = cv2.resize(input_data[0], [int(w*scale), int(h*scale)])
                binary_mask = cv2.resize(
                    input_data[1], [int(w*scale), int(h*scale)], interpolation=0
                )
                input_data = np.stack([img, binary_mask], axis=0)[np.newaxis,...]
            if test_time_augmentation:
                prediction = test_time_aug(
                    input_data, channel_name, nimbus, dataset.normalization_dict,
                    batch_size=batch_size, clip_values=dataset.clip_values
                )
            else:
                prediction = nimbus.predict_segmentation(input_data)
            if not isinstance(prediction, np.ndarray):
                prediction = prediction.cpu().numpy()
            prediction = np.squeeze(prediction)
            if dataset.magnification != nimbus.model_magnification:
                prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
            df = pd.DataFrame(segment_mean(instance_mask, prediction))
            if df_fov.empty:
                df_fov["label"] = df["label"]
                df_fov["fov"] = os.path.basename(fov_path)
            df_fov[channel_name] = df["intensity_mean"]
            if save_predictions:
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = (prediction*255.0).astype(np.uint8)
                io.imwrite(
                    os.path.join(out_fov_path, channel_name + suffix), pred_int,
                    photometric="minisblack",
                    # compress=0, 
                )
        fov_dict_list.append(df_fov)
    cell_table = pd.concat(fov_dict_list, ignore_index=True)
    return cell_table


def calculate_normalization(dataset: MultiplexDataset, quantile: float):
    """Calculates the normalization values for a given ome file

    Args:
        dataset (MultiplexDataset): dataset object
        quantile (float): quantile to use for normalization
    Returns:
        dict: dict with channel names as keys and norm factors  as values
    """
    normalization_values = {}
    for channel in dataset.channels:
        mplex_img = dataset.get_channel(dataset.fovs[0], channel)
        mplex_img = mplex_img.astype(np.float32)
        if np.any(mplex_img):
            foreground = mplex_img[mplex_img > 0]
            normalization_values[channel] = np.quantile(foreground, quantile)
        else:
            normalization_values[channel] = None
    return normalization_values


def prepare_normalization_dict(
        dataset: MultiplexDataset, output_dir: str, quantile: float=0.999, n_subset: int=10,
        n_jobs: int=1, output_name: str="normalization_dict.json"
    ):
    """Prepares the normalization dict for a list of ome.tif fovs

    Args:
        MultiplexDataset (list): list of paths to fovs
        output_dir (str): path to output directory
        quantile (float): quantile to use for normalization
        n_subset (int): number of fovs to use for normalization
        n_jobs (int): number of jobs to use for joblib multiprocessing
        output_name (str): name of output file
    Returns:
        dict: dict with channel names as keys and norm factors  as values
    """
    normalization_dict = {}
    fov_paths = copy(dataset.fov_paths)
    if n_subset is not None:
        random.shuffle(fov_paths)
        fov_paths = fov_paths[:n_subset]
    print("Iterate over fovs...")
    if n_jobs > 1 and len(fov_paths) > 1:
        normalization_values = Parallel(n_jobs=n_jobs)(
            delayed(calculate_normalization)(
                MultiplexDataset(
                    [fov_path], dataset.segmentation_naming_convention, dataset.channels,
                    dataset.suffix, True
                ), quantile)
            for fov_path in fov_paths
        )
    else:
        normalization_values = [
            calculate_normalization(
                MultiplexDataset(
                    [fov_path], dataset.segmentation_naming_convention, dataset.channels,
                    dataset.suffix, True
                ), quantile)
            for fov_path in fov_paths
        ]
    for norm_dict in normalization_values:
        for channel, normalization_value in norm_dict.items():
            if channel not in normalization_dict:
                normalization_dict[channel] = []
            if normalization_value:
                normalization_dict[channel].append(normalization_value)
    if n_jobs > 1:
        get_reusable_executor().shutdown(wait=True)
    for channel in normalization_dict.keys():
        # exclude None and NaN values before averaging
        norm_values = np.array(normalization_dict[channel])
        norm_values = norm_values[~np.isnan(norm_values)]
        norm_values = np.mean(norm_values)
        if np.isnan(norm_values):
            norm_values = 1e-8
        normalization_dict[channel] = float(norm_values)
    # turn numbers to strings and save normalization dict
    normalization_dict_str = {k: str(v) for k, v in normalization_dict.items()}
    with open(os.path.join(output_dir, output_name), 'w') as f:
        json.dump(normalization_dict_str, f)
    return normalization_dict


def prepare_training_data(
        nimbus, dataset: MultiplexDataset, output_dir: str, tile_size: int=512,
        map_size: int=5
        ):
    """Prepares the training data and stores it into lmdb files for fine-tuning Nimbus
    
    Args:
        nimbus (Nimbus): nimbus object
        dataset (MultiplexDataset): dataset object
        output_dir (str): path to output directory
        tile_size (int): size of the training tiles
        map_size (int): size of the lmdb database in gigabytes
    """
    # create lmdb env
    for split, fovs in ((
        "training", dataset.training_fovs),
        ("validation", dataset.validation_fovs)):
        print(f"Preparing {split} data. storing data in {os.path.join(output_dir, split)}")
        env = lmdb.open(
            os.path.join(output_dir, split),
            map_size=map_size*1024**3,
            map_async=True,
            max_dbs=0,
            create=True
        )
        with env.begin(write=True) as txn:
            for fov in tqdm(fovs):
                instance_mask = dataset.get_segmentation(fov)
                for channel_name in dataset.channels:
                    # load data
                    mplex_img = dataset.get_channel_normalized(fov, channel_name)
                    input_data = prepare_input_data(mplex_img, instance_mask)
                    groundtruth = dataset.get_groundtruth(fov, channel_name)
                    # resize data if necessary
                    if dataset.magnification != nimbus.model_magnification:
                        scale = nimbus.model_magnification / dataset.magnification
                        input_data = np.squeeze(input_data)
                        groundtruth = np.squeeze(groundtruth)
                        _, h,w = input_data.shape
                        img = cv2.resize(input_data[0], [int(w*scale), int(h*scale)])
                        binary_mask = cv2.resize(
                            input_data[1], [int(w*scale), int(h*scale)], interpolation=0
                        )
                        input_data = np.stack([img, binary_mask], axis=0) # 2, h, w
                        groundtruth = cv2.resize(
                            groundtruth.astype(np.uint8), [int(w*scale), int(h*scale)],
                            interpolation=0
                        )[np.newaxis, ...] # 1, h, w
                        inst_mask = cv2.resize(
                            instance_mask.astype(np.uint8), [int(w*scale), int(h*scale)],
                            interpolation=0
                        )[np.newaxis, ...]
                    # mirror pad and tile data
                    h, w = input_data.shape[-2:]
                    h_pad = h % tile_size
                    w_pad = w % tile_size
                    input_data = np.pad(input_data, ((0, 0), (0,h_pad), (0,w_pad)), mode="reflect")
                    groundtruth = np.pad(groundtruth, ((0, 0), (0,h_pad), (0,w_pad)), mode="reflect")
                    inst_mask = np.pad(inst_mask, ((0, 0), (0,h_pad), (0,w_pad)), mode="reflect")
                    h, w = input_data.shape[-2:]
                    for i in range(0, h, tile_size):
                        for j in range(0, w, tile_size):
                            input_tile = input_data[..., i:i+tile_size, j:j+tile_size] # 2, h, w
                            gt_tile = groundtruth[..., i:i+tile_size, j:j+tile_size] # 1, h, w
                            inst_tile = inst_mask[..., i:i+tile_size, j:j+tile_size]
                            sample_tile = np.concatenate([input_tile, gt_tile, inst_tile], axis=0)
                            tile_key = f"{fov}_,_{channel_name}_,_{i}_,_{j}"
                            txn.put(tile_key.encode(), sample_tile.tobytes())
        env.close()


class LmdbDataset(torch.utils.data.Dataset):
    """Dataset class for loading data from lmdb files

    Args:
        lmdb_path (str): path to lmdb file
    """
    def __init__(self, lmdb_path: str, tile_size: tuple=(256, 256)):
        with lmdb.open(lmdb_path, readonly=True, max_dbs=0) as env: 
            txn = env.begin()
            # list all keys
            self.keys = [key.decode() for key, _ in txn.cursor()]
        self.length = len(self.keys)
        self.tile_size = tile_size
        self.lmdb_path = lmdb_path

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.length

    def __getitem__(self, idx):
        """Return the sample at the specified index"""
        key = self.keys[idx].encode()
        with lmdb.open(self.lmdb_path, readonly=True, max_dbs=0) as env: 
            txn = env.begin()
            sample = txn.get(key)
            sample = np.frombuffer(sample, dtype=np.float32)
            sample = sample.reshape(4, self.tile_size[0], self.tile_size[1])
            input_data = sample[:2]
            groundtruth = sample[2:3]
            inst_mask = sample[3:]
            return input_data, groundtruth, inst_mask, self.keys[idx]


class InteractiveDataset(object):
    """Dataset for the InteractiveViewer class. This dataset class stores multiple objects of type
    MultiplexedDataset, and allows to select a dataset and use its method for reading fovs and
    channels from it.

    Args:
        datasets (dict): dictionary with dataset names as keys and dataset objects as values
    """
    def __init__(self, datasets: dict):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.dataset = None

    def set_dataset(self, dataset_name: str):
        """Set the active dataset

        Args:
            dataset_name (str): name of the dataset
        """
        self.dataset = self.datasets[dataset_name]
        return self.dataset

    def get_channel(self, fov: str, channel: str):
        """Get a channel from a fov

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: channel image
        """
        return self.dataset.get_channel(fov, channel)

    def get_segmentation(self, fov: str):
        """Get the instance mask for a fov

        Args:
            fov (str): name of a fov
        Returns:
            np.array: instance mask
        """
        return self.dataset.get_segmentation(fov)

    def get_groundtruth(self, fov: str, channel: str):
        """Get the groundtruth for a fov / channel combination

        Args:
            fov (str): name of a fov
            channel (str): channel name
        Returns:
            np.array: groundtruth activity mask (0: negative, 1: positive, 2: ambiguous)
        """
        return self.dataset.get_groundtruth(fov, channel)
