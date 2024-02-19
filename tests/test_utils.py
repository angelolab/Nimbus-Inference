from nimbus_inference.utils import prepare_normalization_dict, calculate_normalization
from nimbus_inference.utils import predict_fovs, predict_ome_fovs, prepare_input_data
from nimbus_inference.utils import test_time_aug as tt_aug
from nimbus_inference.nimbus import Nimbus
from skimage import io
from pyometiff import OMETIFFWriter
import numpy as np
import tempfile
import torch
import json
import os


class MockModel(torch.nn.Module):
    def __init__(self, padding):
        super(MockModel, self).__init__()
        self.padding = padding
        self.fn = torch.nn.Identity()
    
    def forward(self, x):
        return self.fn(x)


def prepare_tif_data(num_samples, temp_dir, selected_markers, random=False, std=1):
    np.random.seed(42)
    fov_paths = []
    inst_paths = []
    deepcell_dir = os.path.join(temp_dir, "deepcell_output")
    os.makedirs(deepcell_dir, exist_ok=True)
    if isinstance(std, (int, float)) or len(std) != len(selected_markers):
        std = [std] * len(selected_markers)
    for i in range(num_samples):
        folder = os.path.join(temp_dir, "fov_" + str(i))
        os.makedirs(folder, exist_ok=True)
        for marker, scale in zip(selected_markers, std):
            if random:
                img = np.random.rand(256, 256) * scale
            else:
                img = np.ones([256, 256])
            io.imsave(
                os.path.join(folder, marker + ".tiff"),
                img,
            )
        inst_path = os.path.join(deepcell_dir, f"fov_{i}_whole_cell.tiff")
        io.imsave(
                inst_path, np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
                ).repeat(64, axis=1).repeat(64, axis=0)
        )
        if folder not in fov_paths:
            fov_paths.append(folder)
            inst_paths.append(inst_path)
    return fov_paths, inst_paths


def prepare_ome_tif_data(num_samples, temp_dir, selected_markers, random=False, std=1):
    np.random.seed(42)
    metadata_dict = {
        "PhysicalSizeX" : "0.88",
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeY" : "0.88",
        "PhysicalSizeYUnit" : "µm",
        "PhysicalSizeZ" : "3.3",
        "PhysicalSizeZUnit" : "µm",
    }

    for i in range(num_samples):
        metadata_dict["Channels"] = {}
        channels = []
        for marker in zip(selected_markers):
            if random:
                img = np.random.rand(256, 256) * std
            else:
                img = np.ones([256, 256])
                channels.append(img)
            metadata_dict["Channels"][marker] = {
                "Name" : marker,
                "SamplesPerPixel": 1,
            }
        channel_data = np.stack(channels, axis=0)
        sample_name = os.path.join(temp_dir, f"fov_{i}.ome.tiff")
        dimension_order = "CYX"
        writer = OMETIFFWriter(
            fpath=sample_name,
            dimension_order=dimension_order,
            array=channel_data,
            metadata=metadata_dict,
            explicit_tiffdata=False)
        writer.write()
    return None


def test_calculate_normalization():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4"], random=True, std=[0.5]
        )
        channel = "CD4"
        channel_path = os.path.join(fov_paths[0], channel + ".tiff")
        channel_out, norm_val = calculate_normalization(channel_path, 0.999)
        # test if we get the correct channel and normalization value
        assert channel_out == channel
        assert np.isclose(norm_val, 0.5, 0.01)


def test_prepare_normalization_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5, 1.0, 1.5, 2.0, 5.0]
        channels = ["CD4", "CD11c", "CD14", "CD56", "CD57"]
        fov_paths, _ = prepare_tif_data(
            num_samples=5, temp_dir=temp_dir, selected_markers=channels, random=True, std=scales
        )
        normalization_dict = prepare_normalization_dict(
            fov_paths, temp_dir, quantile=0.999, n_subset=10, n_jobs=1,
            output_name="normalization_dict.json"
        )
        # test if normalization dict got saved
        assert os.path.exists(os.path.join(temp_dir, "normalization_dict.json"))
        assert normalization_dict == json.load(
            open(os.path.join(temp_dir, "normalization_dict.json"))
        )
        # test if normalization dict is correct
        for channel, scale in zip(channels, scales):
            assert np.isclose(normalization_dict[channel], scale, 0.01)

        # test if multiprocessing yields approximately the same results
        normalization_dict_mp = prepare_normalization_dict(
            fov_paths, temp_dir, quantile=0.999, n_subset=10, n_jobs=2,
            output_name="normalization_dict.json"
        )
        for key in normalization_dict.keys():
            assert np.isclose(normalization_dict[key], normalization_dict_mp[key], 1e-6)


def test_prepare_input_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5]
        channels = ["CD4"]
        fov_paths, inst_paths = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=channels, random=True, std=scales
        )
        mplex_img = io.imread(os.path.join(fov_paths[0], "CD4.tiff"))
        instance_mask = io.imread(inst_paths[0])
        input_data = prepare_input_data(mplex_img, instance_mask)
        # check shape
        assert input_data.shape == (1, 2, 256, 256)
        # check if instance mask got binarized and eroded
        assert np.alltrue(np.unique(input_data[:,1]) == np.array([0, 1]))
        assert np.sum(input_data[:,1]) < np.sum(instance_mask)
        # check if mplex image is the same as before
        assert np.alltrue(input_data[0, 0] == mplex_img)


def test_tt_aug():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")
        channel = "CD4"
        fov_paths, inst_paths = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=[channel]
        )
        output_dir = os.path.join(temp_dir, "nimbus_output")
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir,
        )
        nimbus.prepare_normalization_dict()
        mplex_img = io.imread(os.path.join(fov_paths[0], channel+".tiff"))
        instance_mask = io.imread(inst_paths[0])
        input_data = prepare_input_data(mplex_img, instance_mask)
        nimbus.model = MockModel(padding="reflect")
        pred_map = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=True, flip=True,
            batch_size=32
        )
        # check if we get the correct shape
        assert pred_map.shape == (2, 256, 256)

        pred_map_2 = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=False, flip=True,
            batch_size=32
        )
        pred_map_3 = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=True, flip=False,
            batch_size=32
        )
        pred_map_no_tt_aug = nimbus.predict_segmentation(
            input_data,
            preprocess_kwargs={
                "normalize": True,
                "marker": channel,
                "normalization_dict": nimbus.normalization_dict},
        )
        # check if we get roughly the same results for non augmented and augmented predictions
        assert np.allclose(pred_map, pred_map_no_tt_aug, atol=0.05)
        assert np.allclose(pred_map_2, pred_map_no_tt_aug, atol=0.05)
        assert np.allclose(pred_map_3, pred_map_no_tt_aug, atol=0.05)


def test_predict_fovs():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")

        fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"]
        )
        output_dir = os.path.join(temp_dir, "nimbus_output")
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir,
        )
        output_dir = os.path.join(temp_dir, "nimbus_output")
        nimbus.prepare_normalization_dict()
        cell_table = predict_fovs(
            nimbus=nimbus, fov_paths=fov_paths, output_dir=output_dir,
            normalization_dict=nimbus.normalization_dict,
            segmentation_naming_convention=segmentation_naming_convention, suffix=".tiff",
            save_predictions=False, half_resolution=True,
        )
        # check if we get the correct number of cells
        assert len(cell_table) == 15
        # check if we get the correct columns (fov, label, CD4, CD56)
        assert np.alltrue(
            set(cell_table.columns) == set(["fov", "label", "CD4", "CD56"])
        )
        # check if predictions don't get written to output_dir
        assert not os.path.exists(os.path.join(output_dir, "fov_0", "CD4.tiff"))
        assert not os.path.exists(os.path.join(output_dir, "fov_0", "CD56.tiff"))
        #
        # run again with save_predictions=True and check if predictions get written to output_dir
        cell_table = predict_fovs(
            nimbus=nimbus, fov_paths=fov_paths, output_dir=output_dir,
            normalization_dict=nimbus.normalization_dict,
            segmentation_naming_convention=segmentation_naming_convention, suffix=".tiff",
            save_predictions=True, half_resolution=True,
        )
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD4.tiff"))
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD56.tiff"))
