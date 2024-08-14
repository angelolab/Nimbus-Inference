from tests.test_utils import prepare_ome_tif_data, prepare_tif_data
from pytest_socket import disable_socket, enable_socket
import pytest
import tempfile
from nimbus_inference.utils import MultiplexDataset
from nimbus_inference.nimbus import Nimbus, prep_naming_convention
from nimbus_inference.unet import UNet
from skimage.data import astronaut
from skimage.transform import rescale
import numpy as np
import torch
import os


def test_check_inputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        num_samples = 5
        selected_markers = ["CD45", "CD3", "CD8", "ChyTr"]
        fov_paths, _ = prepare_tif_data(num_samples, temp_dir, selected_markers)
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))
        dataset = MultiplexDataset(fov_paths, naming_convention)
        nimbus = Nimbus(dataset=dataset, output_dir=temp_dir)
        nimbus.check_inputs()


def test_initialize_model():
    dataset = MultiplexDataset(["tests"])
    nimbus = Nimbus(
        dataset, output_dir="",
        input_shape=[512,512], batch_size=4
    )
    nimbus.initialize_model(padding="valid")
    assert isinstance(nimbus.model, UNet)
    assert nimbus.model.padding == "valid"
    nimbus.initialize_model(padding="reflect")
    assert isinstance(nimbus.model, UNet)
    assert nimbus.model.padding == "reflect"
    # test if model gets loaded in offline mode when it was loaded from huggingface hub before
    nimbus.model = None
    disable_socket()
    nimbus.initialize_model(padding="valid")
    assert isinstance(nimbus.model, UNet)


def test_prepare_normalization_dict():
    # test if normalization dict gets prepared and saved, in-depth tests are in inference_test.py
    with tempfile.TemporaryDirectory() as temp_dir:

        num_samples = 5
        selected_markers = ["CD45", "CD3", "CD8", "ChyTr"]
        fov_paths,_ = prepare_tif_data(num_samples, temp_dir, selected_markers)
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))
        dataset = MultiplexDataset(
            fov_paths, naming_convention, include_channels=["CD45", "CD3", "CD8"]
        )
        nimbus = Nimbus(dataset, temp_dir)
        # test if normalization dict gets prepared and saved
        nimbus.prepare_normalization_dict(overwrite=True)
        assert os.path.exists(os.path.join(temp_dir, "normalization_dict.json"))
        assert "ChyTr" not in nimbus.normalization_dict.keys()

        # test if normalization dict gets loaded
        nimbus_2 = Nimbus(dataset, temp_dir)
        nimbus_2.prepare_normalization_dict()
        assert nimbus_2.normalization_dict == nimbus.normalization_dict


def test_tile_input():
    image = torch.rand([1,2,768,768])
    tile_size = (512, 512)
    output_shape = (320,320)
    dataset = MultiplexDataset(["tests"])
    nimbus = Nimbus(MultiplexDataset, output_dir="")
    nimbus.model = lambda x: x[..., 96:-96, 96:-96]
    tiled_input, padding = nimbus._tile_input(image, tile_size, output_shape)
    assert tiled_input.shape == (3,3,1,2,512,512)
    assert padding == [192, 192, 192, 192]


def test_tile_and_stitch():
    # tests _tile_and_stitch which chains _tile_input, model.forward and _stitch_tiles 
    image = rescale(astronaut(), 1.5, channel_axis=-1)
    image = np.moveaxis(image, -1, 0)[np.newaxis, ...]
    nimbus = Nimbus(
        dataset="", output_dir="", input_shape=[512,512], batch_size=4
    )
    # check if tile and stitch works for mock model unequal input and output shape
    # mock model only center crops the input, so that the stitched output is equal to the input
    for s in [41, 89, 96]:
        nimbus.model = lambda x: x[..., s:-s, s:-s]
        out = nimbus._tile_and_stitch(image)
        assert np.all(
            np.isclose(image, out, rtol=1e-4)
        )
    # check if tile and stitch works with the real model
    nimbus.initialize_model(padding="valid")
    image = np.random.rand(1, 2, 768, 768)
    prediction = nimbus._tile_and_stitch(image)
    assert prediction.shape == (1, 1, 768, 768)
    assert prediction.max() <= 1
    assert prediction.min() >= 0        
