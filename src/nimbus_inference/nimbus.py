from alpineer import io_utils, misc_utils
from skimage.util.shape import view_as_windows
import nimbus_inference
from nimbus_inference.utils import (prepare_normalization_dict,
    predict_fovs, nimbus_preprocess, MultiplexDataset
)
from huggingface_hub import hf_hub_download
from nimbus_inference.unet import UNet
from tqdm.autonotebook import tqdm
from pathlib import Path
from torch import nn
from glob import glob
import numpy as np
import torch
import json
import os
import re


def nimbus_preprocess(image, **kwargs):
    """Preprocess input data for Nimbus model.
    Args:
        image: array to be processed
    Returns:
        np.array: processed image array
    """
    output = np.copy(image)
    if len(image.shape) != 4:
        raise ValueError("Image data must be 4D, got image of shape {}".format(image.shape))

    normalize = kwargs.get("normalize", True)
    if normalize:
        marker = kwargs.get("marker", None)
        if re.search(".tiff|.tiff|.png|.jpg|.jpeg", marker, re.IGNORECASE):
            marker = marker.split(".")[0]
        normalization_dict = kwargs.get("normalization_dict", {})
        if marker in normalization_dict.keys():
            norm_factor = normalization_dict[marker]
        else:
            print(
                "Norm_factor not found for marker {}, calculating directly from the image. \
            ".format(
                    marker
                )
            )
            norm_factor = np.quantile(output[:,0,...], 0.999)
        # normalize only marker channel in chan 0 not binary mask in chan 1
        output[:,0,...] /= norm_factor
        # output = output.clip(0, 1)
    return output


def prep_naming_convention(deepcell_output_dir):
    """Prepares the naming convention for the segmentation data
    Args:
        deepcell_output_dir (str): path to directory where segmentation data is saved
    Returns:
        segmentation_naming_convention (function): function that returns the path to the
            segmentation data for a given fov
    """

    def segmentation_naming_convention(fov_path):
        """Prepares the path to the segmentation data for a given fov
        Args:
            fov_path (str): path to fov
        Returns:
            seg_path (str): paths to segmentation fovs
        """
        fov_name = os.path.basename(fov_path).replace(".ome.tiff", "")
        return os.path.join(deepcell_output_dir, fov_name + "_whole_cell.tiff")

    return segmentation_naming_convention


class Nimbus(nn.Module):
    """Nimbus application class for predicting marker activity for cells in multiplexed images."""

    def __init__(
        self, dataset: MultiplexDataset, output_dir: str, save_predictions: bool=True,
        half_resolution: bool=True, batch_size: int=4, test_time_aug: bool=True,
        input_shape: list=[1024, 1024], device: str="auto",
    ):
        """Initializes a Nimbus Application.
        Args:
            dataset (MultiplexDataset): Path to directory containing fovs.
            output_dir (str): Path to directory to save output.
            save_predictions (bool): Whether to save predictions.
            half_resolution (bool): Whether to run model on half resolution images.
            batch_size (int): Batch size for model inference.
            test_time_aug (bool): Whether to use test time augmentation.
            input_shape (list): Shape of input images.
            suffix (str): Suffix of images to load.
            device (str): Device to run model on, either "auto" (either "mps" or "cuda"
            , with "cpu" as a fallback), "cpu", "cuda", or "mps". Defaults to "auto".
        """
        super(Nimbus, self).__init__()
        self.dataset = dataset
        self.output_dir = output_dir
        self.half_resolution = half_resolution
        self.save_predictions = save_predictions
        self.batch_size = batch_size
        self.checked_inputs = False
        self.test_time_aug = test_time_aug
        self.input_shape = input_shape
        if self.output_dir != "":
            os.makedirs(self.output_dir, exist_ok=True)

        if device == "auto":
          if torch.backends.mps.is_available():
            self.device = torch.device("mps")
          elif torch.cuda.is_available():
            self.device = torch.device("cuda")
          else:
            self.device = torch.device("cpu")
        else:
            misc_utils.verify_in_list(device=[device], valid_devices=["cpu", "cuda", "mps"])
            self.device = torch.device(device)

    def check_inputs(self):
        """check inputs for Nimbus model"""
        # check if output_dir exists
        io_utils.validate_paths([self.output_dir])

        self.checked_inputs = True
        print("All inputs are valid.")

    def initialize_model(self, padding="reflect"):
        """Initializes the model and load weights.
        Args:
            padding (str): Padding mode for model, either "reflect" or "valid".
        """
        model = UNet(num_classes=1, padding=padding)
        # relative path to weights
        path = os.path.dirname(nimbus_inference.__file__)
        path = Path(path).resolve()
        self.checkpoint_path = os.path.join(
            path,
            "assets",
            "resUnet_baseline_hickey_tonic_dec_mskc_mskp_2_channel_halfres_512_bs32_cw_0.8.pt"
        )
        if not os.path.exists(self.checkpoint_path):
            local_dir = os.path.join(path, "assets")
            os.makedirs(local_dir, exist_ok=True)
            print("Downloading weights from Hugging Face Hub...")
            self.checkpoint_path = hf_hub_download(
                repo_id="JLrumberger/Nimbus-Inference",
                filename="resUnet_baseline_hickey_tonic_dec_mskc_mskp_2_channel_halfres_512_bs32_cw_0.8.pt",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        model.load_state_dict(torch.load(self.checkpoint_path))
        print("Loaded weights from {}".format(self.checkpoint_path))
        self.model = model.to(self.device).eval()

    def prepare_normalization_dict(
        self, quantile=0.999, n_subset=10, multiprocessing=False, overwrite=False,
    ):
        """Load or prepare and save normalization dictionary for Nimbus model.
        Args:
            quantile (float): Quantile to use for normalization.
            n_subset (int): Number of fovs to use for normalization.
            multiprocessing (bool): Whether to use multiprocessing.
            overwrite (bool): Whether to overwrite existing normalization dict.
        Returns:
            dict: Dictionary of normalization factors.
        """
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")
        if os.path.exists(self.normalization_dict_path) and not overwrite:
            self.normalization_dict = json.load(open(self.normalization_dict_path))
        else:
            n_jobs = os.cpu_count() if multiprocessing else 1
            self.normalization_dict = prepare_normalization_dict(
                self.dataset, self.output_dir, quantile, n_subset,
                n_jobs
            )

    def predict_fovs(self):
        """Predicts cell classification for input data.
        Returns:
            np.array: Predicted cell classification.
        """
        if self.checked_inputs == False:
            self.check_inputs()
        if not hasattr(self, "normalization_dict"):
            self.prepare_normalization_dict()
        # check if GPU is available
        gpus = torch.cuda.device_count()
        print("Available GPUs: ", gpus)
        print("Predictions will be saved in {}".format(self.output_dir))
        print("Iterating through fovs will take a while...")
        self.cell_table = predict_fovs(
            nimbus=self, dataset=self.dataset, output_dir=self.output_dir,
            normalization_dict=self.normalization_dict, save_predictions=self.save_predictions,
            half_resolution=self.half_resolution, batch_size=self.batch_size,
            test_time_augmentation=self.test_time_aug, suffix=self.dataset.suffix,
        )
        self.cell_table.to_csv(os.path.join(self.output_dir, "nimbus_cell_table.csv"), index=False)
        return self.cell_table

    def predict_segmentation(self, input_data, preprocess_kwargs):
        """Predicts segmentation for input data.
        Args:
            input_data (np.array): Input data to predict segmentation for.
            preprocess_kwargs (dict): Keyword arguments for preprocessing.
            batch_size (int): Batch size for prediction.
        Returns:
            np.array: Predicted segmentation.
        """
        input_data = nimbus_preprocess(input_data, **preprocess_kwargs)
        if np.all(np.greater_equal(self.input_shape, input_data.shape[-2:])):
            if not hasattr(self, "model") or self.model.padding != "reflect":
                self.initialize_model(padding="reflect")
            with torch.no_grad():
                if not isinstance(input_data, torch.Tensor):
                    input_data = torch.tensor(input_data).float()
                input_data = input_data.to(self.device)
                prediction = self.model(input_data)
                prediction = prediction.cpu()
        else:
            if not hasattr(self, "model") or self.model.padding != "valid":
                self.initialize_model(padding="valid")
            prediction = self._tile_and_stitch(input_data)
        return prediction

    def _tile_and_stitch(self, input_data):
        """Predicts segmentation for input data using tile and stitch method.
        Args:
            input_data (np.array): Input data to predict segmentation for.
            batch_size (int): Batch size for prediction.
        Returns:
            np.array: Predicted segmentation.
        """
        with torch.no_grad():
            output_shape = self.model(torch.rand(1, 2, *self.input_shape).to(self.device)).shape[-2:]
        # f^dl crop to have perfect shift equivariance inference
        self.crop_by = np.array(output_shape) % 2 ** 5
        output_shape = output_shape - self.crop_by
        tiled_input, padding = self._tile_input(
            input_data, tile_size=self.input_shape, output_shape=output_shape
        )
        shape_diff = (self.input_shape - output_shape) // 2
        padding = [
            padding[0] - shape_diff[0], padding[1] - shape_diff[0],
            padding[2] - shape_diff[1], padding[3] - shape_diff[1],
        ]
        h_t, h_w, b, c = tiled_input.shape[:4]
        tiled_input = tiled_input.reshape(
            h_t * h_w * b, c, *self.input_shape
        )  # h_t,w_t,c,h,w -> h_t*w_t,c,h,w
        # predict tiles
        prediction = []
        for i in tqdm(range(0, len(tiled_input), self.batch_size)):
            batch = torch.from_numpy(tiled_input[i : i + self.batch_size]).float()
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.model(batch).cpu().numpy()
                # crop pred
                if self.crop_by.any():
                    pred = pred[
                        ...,
                        self.crop_by[0]//2 : -self.crop_by[0]//2,
                        self.crop_by[1]//2 : -self.crop_by[1]//2,
                    ]
                prediction += [pred]
        prediction = np.concatenate(prediction)  # h_t*w_t,c,h,w
        prediction = prediction.reshape(h_t, h_w, b, *prediction.shape[1:])  # h_t,w_t,b,c,h,w
        # stitch tiles
        prediction = self._stitch_tiles(prediction, padding)
        return prediction

    def _tile_input(self, image, tile_size, output_shape, pad_mode="reflect"):
        """Tiles input image for model inference.
        Args:
            image (np.array): Image to tile b,c,h,w.
            tile_size (list): Size of input tiles.
            output_shape (list): Shape of model output.
            pad_mode (str): Padding mode for tiling.
        Returns:
            list: List of tiled images.
        """
        # pad image to be divisible by tile size
        pad_h0, pad_w0 = np.array(tile_size) - (
            np.array(image.shape[-2:]) % np.array(output_shape)
        )
        pad_h1, pad_w1 = pad_h0 // 2, pad_w0 // 2
        pad_h0, pad_w0 = pad_h0 - pad_h1, pad_w0 - pad_w1
        image = np.pad(image, ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)), mode=pad_mode)
        b, c = image.shape[:2]
        # tile image
        view = np.squeeze(
            view_as_windows(image, [b, c] + list(tile_size), step=[b, c] + list(output_shape)),
            axis=(0,1)
        )
        # h_t,w_t,b,c,h,w
        padding = [pad_h0, pad_h1, pad_w0, pad_w1]
        return view, padding

    def _stitch_tiles(self, tiles, padding):
        """Stitches tiles to reconstruct full image.
        Args:
            tiles (np.array): Tiled predictions n_tiles x c x h x w.
            input_shape (list): Shape of input image.
        Returns:
            np.array: Reconstructed image.
        """
        # stitch tiles
        h_t, w_t, b, c, h, w = tiles.shape
        stitched = np.zeros((b, c, h_t * h, w_t * w))
        for i in range(h_t):
            for j in range(w_t):
                for b_ in range(b):
                    stitched[b_, :, i * h : (i + 1) * h, j * w : (j + 1) * w] = tiles[i, j, b_]
        # remove padding
        stitched = stitched[:, :, padding[0] : -padding[1], padding[2] : -padding[3]]
        return stitched
