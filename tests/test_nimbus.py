from tests.test_utils import prepare_ome_tif_data, prepare_tif_data
from pytest_socket import disable_socket, enable_socket
import pytest
import tempfile
from nimbus_inference.utils import MultiplexDataset
from nimbus_inference.nimbus import Nimbus, prep_naming_convention
from nimbus_inference.unet import UNet
from skimage.data import astronaut
import nimbus_inference
from pathlib import Path
from skimage.transform import rescale
import numpy as np
import torch
import os

class MockModel(torch.nn.Module):
    def __init__(self, crop_size=96):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[..., self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]


def test_check_inputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        num_samples = 5
        selected_markers = ["CD45", "CD3", "CD8", "ChyTr"]
        fov_paths, _ = prepare_tif_data(num_samples, temp_dir, selected_markers)
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))
        dataset = MultiplexDataset(fov_paths, naming_convention)
        nimbus = Nimbus(dataset=dataset, output_dir=temp_dir)
        nimbus.check_inputs()


def test_load_latest_checkpoint():
    dataset = MultiplexDataset(["tests"])
    nimbus = Nimbus(
        dataset, output_dir="",
        input_shape=[512,512], batch_size=4
    )
    nimbus.load_latest_checkpoint(padding="valid")
    assert isinstance(nimbus.model, UNet)
    assert nimbus.model.padding == "valid"
    nimbus.load_latest_checkpoint(padding="reflect")
    assert isinstance(nimbus.model, UNet)
    assert nimbus.model.padding == "reflect"
    # test if model gets loaded in offline mode when it was loaded from huggingface hub before
    nimbus.model = None
    disable_socket()
    nimbus.load_latest_checkpoint(padding="valid")
    assert isinstance(nimbus.model, UNet)


def test_tile_input():
    image = torch.rand([1,2,768,768])
    tile_size = (512, 512)
    output_shape = (320,320)
    dataset = MultiplexDataset(["tests"])
    nimbus = Nimbus(MultiplexDataset, output_dir="")
    nimbus.model = MockModel(96) 
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
        nimbus.model = MockModel(s)
        out = nimbus._tile_and_stitch(image)
        assert np.all(
            np.isclose(image, out, rtol=1e-4)
        )
    # check if tile and stitch works with the real model
    nimbus.load_latest_checkpoint(padding="valid")
    image = np.random.rand(1, 2, 768, 768)
    prediction = nimbus._tile_and_stitch(image)
    assert prediction.shape == (1, 1, 768, 768)
    assert prediction.max() <= 1
    assert prediction.min() >= 0        


def test_load_local_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup basic dataset
        dataset = MultiplexDataset(["tests"])
        nimbus = Nimbus(dataset, output_dir="")

        # Create mock checkpoint
        path = os.path.dirname(nimbus_inference.__file__)
        path = Path(path).resolve()
        local_dir = os.path.join(path, "assets")
        os.makedirs(local_dir, exist_ok=True)
        mock_checkpoint = "V1.pt"
        checkpoint_path = os.path.join(local_dir, mock_checkpoint)
        
        # Save mock model state
        mock_model = UNet(num_classes=1, padding="reflect")
        torch.save(mock_model.state_dict(), checkpoint_path)

        # Test successful loading with reflect padding
        nimbus.load_local_checkpoint(mock_checkpoint, padding="reflect")
        assert isinstance(nimbus.model, UNet)
        assert nimbus.model.padding == "reflect"
        assert nimbus.checkpoint_path == checkpoint_path

        # Test successful loading with valid padding
        nimbus.load_local_checkpoint(mock_checkpoint, padding="valid")
        assert isinstance(nimbus.model, UNet)
        assert nimbus.model.padding == "valid"

        # Test invalid checkpoint name
        with pytest.raises(ValueError) as exc_info:
            nimbus.load_local_checkpoint("invalid_checkpoint.pt")
        assert "not found in local checkpoints" in str(exc_info.value)

        # Test model is in eval mode
        assert not nimbus.model.training

        # Cleanup
        os.remove(checkpoint_path)


def test_list_checkpoints():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup basic dataset
        dataset = MultiplexDataset(["tests"])
        nimbus = Nimbus(dataset, output_dir=temp_dir)

        # Create mock checkpoints
        path = os.path.dirname(nimbus_inference.__file__)
        path = Path(path).resolve()
        local_dir = os.path.join(path, "assets")
        os.makedirs(local_dir, exist_ok=True)
        known_checkpoint = "V1.pt"

        # List checkpoints
        checkpoints = nimbus.list_checkpoints()
        assert known_checkpoint in checkpoints
