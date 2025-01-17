from nimbus_inference.utils import (prepare_normalization_dict, calculate_normalization,
predict_fovs, prepare_input_data, MultiplexDataset, LazyOMETIFFReader,
_handle_qupath_segmentation_map, prepare_training_data, LmdbDataset)
from nimbus_inference.utils import test_time_aug as tt_aug
from nimbus_inference.nimbus import Nimbus
from skimage import io
from pyometiff import OMETIFFWriter
import pytest
import numpy as np
import pandas as pd
import tempfile
import torch
import json
import lmdb
import os


class MockModel(torch.nn.Module):
    def __init__(self, padding):
        super(MockModel, self).__init__()
        self.padding = padding
        self.param = torch.nn.Parameter(torch.tensor([1.0]))
        self.fn = torch.nn.Identity()
        self.device = torch.device("cpu")
    
    def forward(self, x):
        x = x.mean(1).unsqueeze(1)
        return torch.nn.functional.sigmoid(self.fn(x) * self.param)


def prepare_tif_data(
        num_samples, temp_dir, selected_markers, random=False, std=1, shape=(256, 256),
        image_dtype=np.float32, instance_dtype=np.uint16, qupath_seg=False
    ):
    np.random.seed(42)
    fov_paths = []
    inst_paths = []
    deepcell_dir = os.path.join(temp_dir, "deepcell_output")
    os.makedirs(deepcell_dir, exist_ok=True)
    if isinstance(std, (int, float)) or len(std) != len(selected_markers):
        std = [std] * len(selected_markers)
    df_list = []
    for i in range(num_samples):
        folder = os.path.join(temp_dir, "fov_" + str(i))
        os.makedirs(folder, exist_ok=True)
        for marker, scale in zip(selected_markers, std):
            if random:
                img = np.random.rand(*shape) * scale
            else:
                img = np.ones(shape)
            io.imsave(
                os.path.join(folder, marker + ".tiff"),
                img.astype(image_dtype),
            )
            df_list.append(pd.DataFrame({
                        "fov": [f"fov_{i}"] * 15,
                        "channel": [marker] * 15,
                        "cell_id": np.arange(15)+1,
                        "activity": np.random.randint(0, 2, 15),
                    }))
        inst_path = os.path.join(deepcell_dir, f"fov_{i}_whole_cell.tiff")
        instance_mask = np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
                ).repeat(shape[1]//4, axis=1).repeat(shape[0]//4, axis=0).astype(instance_dtype)
        if qupath_seg:
            instance_mask = np.stack([instance_mask] * 3, axis=-1)
        io.imsave(
                inst_path, instance_mask
        )
        if folder not in fov_paths:
            fov_paths.append(folder)
            inst_paths.append(inst_path)
    groundtruth_df = pd.concat(df_list)
    gt_path = os.path.join(temp_dir, "groundtruth_df.csv")
    groundtruth_df.to_csv(gt_path, index=False)
    # add ds_store file to test if it gets ignored
    ds_store_paths = [
        os.path.join(temp_dir, ".DS_Store"),
        os.path.join(temp_dir, "fov_0", ".DS_Store"),
        os.path.join(temp_dir, "deepcell_output", ".DS_Store"),
    ]
    for ds_store in ds_store_paths:
        with open(ds_store, "w") as f:
            f.write("test")
    return fov_paths, inst_paths


def prepare_ome_tif_data(
        num_samples, temp_dir, selected_markers, random=False, std=1, shape=(256, 256), 
    ):
    np.random.seed(42)
    metadata_dict = {
        "SizeX" : shape[0],
        "SizeY" : shape[1],
        "SizeC" : len(selected_markers) + 3,
        "PhysicalSizeX" : 0.5,
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeY" : 0.5,
        "PhysicalSizeYUnit" : "µm",
    }
    fov_paths = []
    inst_paths = []
    if isinstance(std, (int, float)) or len(std) != len(selected_markers):
        std = [std] * len(selected_markers)
    df_list = []
    for i in range(num_samples):
        metadata_dict["Channels"] = {}
        channels = []
        for j, (marker, s) in enumerate(zip(selected_markers, std)):
            if random:
                img = np.random.rand(*shape) * s
            else:
                img = np.ones(shape)
            channels.append(img)
            metadata_dict["Channels"][marker] = {
                "Name" : marker,
                "ID": str(j),
                "SamplesPerPixel": 1,
            }
            df_list.append(pd.DataFrame({
                        "fov": [f"fov_{i}"] * 15,
                        "channel": [marker] * 15,
                        "cell_id": np.arange(15)+1,
                        "activity": np.random.randint(0, 2, 15),
                    }))
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
        deepcell_dir = os.path.join(temp_dir, "deepcell_output")
        os.makedirs(deepcell_dir, exist_ok=True)
        inst_path = os.path.join(deepcell_dir, f"fov_{i}_whole_cell.tiff")
        io.imsave(
                inst_path, np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
                ).repeat(shape[1]//4, axis=1).repeat(shape[0]//4, axis=0)
        )
        fov_paths.append(sample_name)
        inst_paths.append(inst_path)
    groundtruth_df = pd.concat(df_list)
    gt_path = os.path.join(temp_dir, "groundtruth_df.csv")
    groundtruth_df.to_csv(gt_path, index=False)
    # add ds_store file to test if it gets ignored
    ds_store_paths = [
        os.path.join(temp_dir, ".DS_Store"),
        os.path.join(temp_dir, "deepcell_output", ".DS_Store"),
    ]
    for ds_store in ds_store_paths:
        with open(ds_store, "w") as f:
            f.write("test")
    return fov_paths, inst_paths


def test_calculate_normalization():
    with tempfile.TemporaryDirectory() as temp_dir:
        # test for single channel data
        tif_fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4"], random=True, std=[0.5]
        )
        channel = "CD4"
        tif_dataset = MultiplexDataset(tif_fov_paths, suffix=".tiff", output_dir=temp_dir)
        ome_fov_paths, _ = prepare_ome_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"], random=True, std=[0.5]
        )
        ome_dataset = MultiplexDataset(ome_fov_paths, suffix=".ome.tiff", output_dir=temp_dir)
        for dataset in [tif_dataset, ome_dataset]:
            norm_dict = calculate_normalization(dataset, 0.999)
            channel_out, norm_val = list(norm_dict.items())[0]
            # test if we get the correct channel and normalization value
            assert channel_out == channel
            assert np.isclose(norm_val, 0.5, 0.01)


def test_prepare_normalization_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5, 1.0, 1.5, 2.0, 5.0]
        channels = ["CD4", "CD11c", "CD14", "CD56", "CD57"]
        tif_fov_paths, _ = prepare_tif_data(
            num_samples=5, temp_dir=temp_dir, selected_markers=channels, random=True, std=scales
        )
        tif_dataset = MultiplexDataset(tif_fov_paths, suffix=".tiff", output_dir=temp_dir)
        
        # test if everything works for multi channel data
        ome_fov_paths, _ = prepare_ome_tif_data(
            num_samples=5, temp_dir=temp_dir, selected_markers=channels, random=True, std=scales
        )
        ome_dataset = MultiplexDataset(ome_fov_paths, suffix=".ome.tiff", output_dir=temp_dir)
        for dataset in [tif_dataset, ome_dataset]:
            normalization_dict = prepare_normalization_dict(
                dataset, temp_dir, quantile=0.999, n_subset=10, n_jobs=1,
                output_name="normalization_dict.json"
            )
            # test if normalization dict got saved
            assert os.path.exists(os.path.join(temp_dir, "normalization_dict.json"))
            normalization_dict_loaded = json.load(
                open(os.path.join(temp_dir, "normalization_dict.json"))
            )
            normalization_dict_loaded = {k: float(v) for k, v in normalization_dict_loaded.items()}
            assert normalization_dict == normalization_dict_loaded
            # test if normalization dict is correct
            for channel, scale in zip(channels, scales):
                assert np.isclose(normalization_dict[channel], scale, 0.01)

            # test if multiprocessing yields approximately the same results
            normalization_dict_mp = prepare_normalization_dict(
                dataset, temp_dir, quantile=0.999, n_subset=10, n_jobs=2,
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
        assert np.alltrue(input_data[0, 0] == mplex_img.astype(np.float32))


def test_tt_aug():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")
        channel = "CD4"
        fov_paths, inst_paths = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=[channel], shape=(512, 256)
        )
        output_dir = os.path.join(temp_dir, "nimbus_output")
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff", output_dir=temp_dir
        )
        dataset.prepare_normalization_dict()
        nimbus = Nimbus(dataset, output_dir)
        mplex_img = io.imread(os.path.join(fov_paths[0], channel+".tiff"))
        instance_mask = io.imread(inst_paths[0])
        input_data = prepare_input_data(mplex_img, instance_mask)
        nimbus.model = MockModel(padding="reflect").to(nimbus.device)
        pred_map = tt_aug(
            input_data, channel, nimbus, dataset.normalization_dict, rotate=True, flip=True,
            batch_size=32
        )
        # check if we get the correct shape
        assert pred_map.shape == (512, 256)

        pred_map_2 = tt_aug(
            input_data, channel, nimbus, dataset.normalization_dict, rotate=False, flip=True,
            batch_size=32
        )
        pred_map_3 = tt_aug(
            input_data, channel, nimbus, dataset.normalization_dict, rotate=True, flip=False,
            batch_size=32
        )
        pred_map_no_tt_aug = nimbus.predict_segmentation(input_data)
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
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"], shape=(512, 256),
            instance_dtype=np.float32
        )
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff", output_dir=temp_dir
        )
        dataset.prepare_normalization_dict()
        output_dir = os.path.join(temp_dir, "nimbus_output")
        nimbus = Nimbus(dataset, output_dir)
        cell_table = predict_fovs(
            nimbus=nimbus, dataset=dataset, output_dir=output_dir,
            suffix=".tiff", save_predictions=False, test_time_augmentation=False
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
            nimbus=nimbus, dataset=dataset, output_dir=output_dir, suffix=".tiff",
            save_predictions=True, test_time_augmentation=False
        )
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD4.tiff"))
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD56.tiff"))

        # check with qupath segmentation map
        fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"], shape=(512, 256),
            instance_dtype=np.uint16, qupath_seg=True)
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff", output_dir=temp_dir
        )
        dataset.prepare_normalization_dict()
        output_dir = os.path.join(temp_dir, "nimbus_output_qupath")
        nimbus = Nimbus(dataset, output_dir)
        cell_table = predict_fovs(
            nimbus=nimbus, dataset=dataset, output_dir=output_dir, suffix=".tiff",
            save_predictions=False, test_time_augmentation=False
        )
        # check if we get the correct number of cells
        assert len(cell_table) == 15
        # check if we get the correct columns (fov, label, CD4, CD56)
        assert np.alltrue(
            set(cell_table.columns) == set(["fov", "label", "CD4", "CD56"])
        )


def test_LazyOMETIFFReader():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_ome_tif_data(
        num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"]
        )
        reader = LazyOMETIFFReader(fov_paths[0])
        assert hasattr(reader, "metadata")
        assert reader.channels == ["CD4", "CD56"]
        cd4_channel = reader.get_channel("CD4")
        cd56_channel = reader.get_channel("CD56")
        assert cd4_channel.shape == (256, 256)
        assert cd56_channel.shape == (256, 256)


def test_handle_qupath_segmentation_map():
    # generate a qupath RGB segmentation map
    instance_mask = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    ).repeat(64, axis=1).repeat(64, axis=0).astype(np.float32)
    instance_mask = instance_mask / (255**2)
    instance_mask = np.stack([instance_mask] + [np.zeros_like(instance_mask)]*2, axis=-1)
    instance_mask_handled = _handle_qupath_segmentation_map(instance_mask)
    assert instance_mask_handled.shape == (256, 256)
    assert np.alltrue(np.unique(instance_mask_handled) == np.array(list(range(0,16))))


def test_MultiplexDataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            fov_ = fov_.split(".")[0]
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")

        fov_paths, _ = prepare_ome_tif_data(
        num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
        shape=(512, 256)
        )
        groundtruth_df = pd.read_csv(os.path.join(temp_dir, "groundtruth_df.csv"))
        # check if check inputs raises error when inputs are incorrect
        with pytest.raises(FileNotFoundError):
            dataset_ome = MultiplexDataset(
                ["abc"], segmentation_naming_convention, suffix=".ome.tiff", output_dir=temp_dir
            )
        # check if we get the correct channels and fov_paths
        dataset_ome = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".ome.tiff",
            groundtruth_df=groundtruth_df, output_dir=temp_dir
        )
        assert len(dataset_ome) == 1
        assert set(dataset_ome.channels) == set(["CD4", "CD56"])
        assert dataset_ome.fov_paths == fov_paths
        assert dataset_ome.multi_channel == True
        cd4_channel = io.imread(fov_paths[0])[0]
        cd4_channel_ = dataset_ome.get_channel(fov="fov_0", channel="CD4")
        assert np.alltrue(cd4_channel == cd4_channel_)
        fov_0_seg = io.imread(segmentation_naming_convention(fov_paths[0]))
        fov_0_seg_ = dataset_ome.get_segmentation(fov="fov_0")
        assert np.alltrue(fov_0_seg == fov_0_seg_)
        
        # test everything again with single channel and float dtypes
        fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
            shape=(512, 256), image_dtype=np.float32, instance_dtype=np.float32
        )
        cd4_channel = io.imread(os.path.join(fov_paths[0], "CD4.tiff"))
        fov_0_seg = io.imread(segmentation_naming_convention(fov_paths[0]))
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff",
            groundtruth_df=groundtruth_df, output_dir=temp_dir
        )
        assert len(dataset) == 1
        assert set(dataset.channels) == set(["CD4", "CD56"])
        assert dataset.fov_paths == fov_paths
        assert dataset.multi_channel == False
        cd4_channel_ = dataset.get_channel(fov="fov_0", channel="CD4")
        assert np.alltrue(cd4_channel == cd4_channel_)
        fov_0_seg_ = dataset.get_segmentation(fov="fov_0")
        assert np.alltrue(fov_0_seg == fov_0_seg_)

        for dset in [dataset, dataset_ome]:
            # test if we can get the groundtruth_df
            fov_0_cd4_gt = np.squeeze(dset.get_groundtruth(fov="fov_0", channel="CD4"))
            df = {"fov": ["fov_0"]*15, "channel": ["CD4"]*15, "cell_id": [], "activity": []}
            for i in np.unique(fov_0_seg)[1:]:
                np.mean(fov_0_cd4_gt[fov_0_seg == i])
                df["cell_id"].append(i)
                df["activity"].append(int(np.mean(fov_0_cd4_gt[fov_0_seg == i])))
            df = pd.DataFrame(df)
            subset_df = groundtruth_df[
                (groundtruth_df["fov"] == "fov_0") & (groundtruth_df["channel"] == "CD4")
            ]
            assert np.alltrue(df == subset_df)
        
        # test everything again with single channel and uint16 dtypes
        fov_paths, _ = prepare_tif_data(
            num_samples=1, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
            shape=(512, 256), image_dtype=np.uint16, instance_dtype=np.uint16
        )
        cd4_channel = io.imread(os.path.join(fov_paths[0], "CD4.tiff"))
        fov_0_seg = io.imread(segmentation_naming_convention(fov_paths[0]))
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff",
            groundtruth_df=groundtruth_df, output_dir=temp_dir
        )
        assert len(dataset) == 1
        assert set(dataset.channels) == set(["CD4", "CD56"])
        assert dataset.fov_paths == fov_paths
        assert dataset.multi_channel == False
        cd4_channel_ = dataset.get_channel(fov="fov_0", channel="CD4")
        assert np.alltrue(cd4_channel == cd4_channel_)
        fov_0_seg_ = dataset.get_segmentation(fov="fov_0")
        assert np.alltrue(fov_0_seg == fov_0_seg_)            


def test_prepare_normalization_dict():
    # test if normalization dict gets prepared and saved, in-depth tests are in inference_test.py
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            fov_ = fov_.split(".")[0]
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")

        num_samples = 5
        selected_markers = ["CD45", "CD3", "CD8", "ChyTr"]
        fov_paths,_ = prepare_tif_data(num_samples, temp_dir, selected_markers)
        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, include_channels=["CD45", "CD3", "CD8"],
            output_dir=temp_dir
        )
        # test if normalization dict gets prepared and saved
        dataset.prepare_normalization_dict(overwrite=True)
        assert os.path.exists(os.path.join(temp_dir, "normalization_dict.json"))
        assert "ChyTr" not in dataset.normalization_dict.keys()

        # test if normalization dict gets loaded
        dataset_2 = MultiplexDataset(
            fov_paths, segmentation_naming_convention, include_channels=["CD45", "CD3", "CD8"],
            output_dir=temp_dir
        )
        dataset_2.prepare_normalization_dict(overwrite=False)
        assert dataset_2.normalization_dict == dataset.normalization_dict


def test_prepare_training_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            fov_ = fov_.split(".")[0]
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")
        # Prepare mock data
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
            shape=(512, 256)
        )
        groundtruth_df = pd.read_csv(os.path.join(temp_dir, "groundtruth_df.csv"))

        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff",
            groundtruth_df=groundtruth_df,
            output_dir=temp_dir
        )
        dataset.prepare_normalization_dict()
        nimbus = Nimbus(dataset, output_dir=temp_dir)

        # Create output directory
        output_dir = os.path.join(temp_dir, "lmdb_output")
        os.makedirs(output_dir, exist_ok=True)

        # Run prepare_training_data function
        prepare_training_data(nimbus, dataset, output_dir, tile_size=256, map_size=1)

        # Check if LMDB files are created
        for split in ["training", "validation"]:
            lmdb_path = os.path.join(output_dir, split)
            assert os.path.exists(lmdb_path), f"{lmdb_path} does not exist"
            
            # Check if LMDB environment can be opened
            env = lmdb.open(lmdb_path, readonly=True)
            with env.begin() as txn:
                cursor = txn.cursor()
                assert cursor.first(), f"No data found in {lmdb_path}"

        # Check if data is correctly stored in LMDB
        for split in ["training", "validation"]:
            lmdb_path = os.path.join(output_dir, split)
            env = lmdb.open(lmdb_path, readonly=True)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    sample = np.frombuffer(value, dtype=np.float32)
                    sample = sample.reshape(4, 256, 256)  # Full shape is [3, tile_size, tile_size]
                    input_data = sample[:2]
                    groundtruth = sample[2:3]
                    inst_mask = sample[3:4]
                    # Check if input_data and groundtruth have the correct shape
                    assert input_data.shape == (2, 256, 256)
                    assert groundtruth.shape == (1, 256, 256)
                    assert inst_mask.shape == (1, 256, 256)
            env.close()


def test_LmdbDataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            temp_dir_, fov_ = os.path.split(fov_path)
            fov_ = fov_.split(".")[0]
            return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")
        # Prepare mock data
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
            shape=(512, 256)
        )
        groundtruth_df = pd.read_csv(os.path.join(temp_dir, "groundtruth_df.csv"))

        dataset = MultiplexDataset(
            fov_paths, segmentation_naming_convention, suffix=".tiff",
            groundtruth_df=groundtruth_df,
            output_dir=temp_dir
        )
        dataset.prepare_normalization_dict()
        nimbus = Nimbus(dataset, output_dir=temp_dir)

        # Create output directory
        output_dir = os.path.join(temp_dir, "lmdb_output")
        os.makedirs(output_dir, exist_ok=True)

        # Run prepare_training_data function
        prepare_training_data(nimbus, dataset, output_dir, tile_size=256, map_size=1)

        lmdb_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        
        # Test data retrieval and shapes
        for input_data, groundtruth, inst_mask, key in lmdb_dataset:
            
            # Check shapes
            assert input_data.shape == (2, 256, 256)
            assert groundtruth.shape == (1, 256, 256)
            assert inst_mask.shape == (1, 256, 256)
            assert isinstance(key, str)
            
            # Check content matches what we stored
            assert np.isin(input_data, [0, 1]).all()
            assert np.isin(groundtruth, [0, 1, 2]).all()
            assert np.isin(inst_mask, np.arange(16)).all()
            assert key in lmdb_dataset.keys
