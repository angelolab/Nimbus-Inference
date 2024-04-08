from nimbus_inference.viewer_widget import NimbusViewer
from nimbus_inference.nimbus import Nimbus, prep_naming_convention
from nimbus_inference.utils import MultiplexDataset
from tests.test_utils import prepare_ome_tif_data, prepare_tif_data
import numpy as np
import tempfile
import os


def test_NimbusViewer():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        dataset = MultiplexDataset(fov_paths)
        viewer_widget = NimbusViewer(dataset, temp_dir)
        assert isinstance(viewer_widget, NimbusViewer)


def test_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        dataset = MultiplexDataset(fov_paths)
        viewer_widget = NimbusViewer(dataset, temp_dir)
        path_dict = {
            "red": os.path.join(temp_dir, "fov_0", "CD4.tiff"),
            "green": os.path.join(temp_dir, "fov_0", "CD11c.tiff"),
        }
        composite_image = viewer_widget.create_composite_image(path_dict)
        assert isinstance(composite_image, np.ndarray)
        assert composite_image.shape == (256, 256, 3)

        path_dict["blue"] = os.path.join(temp_dir, "fov_0", "CD56.tiff")
        composite_image = viewer_widget.create_composite_image(path_dict)
        assert composite_image.shape == (256, 256, 3)


def test_create_composite_from_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        dataset = MultiplexDataset(fov_paths)
        viewer_widget = NimbusViewer(dataset, temp_dir)
        path_dict = {
            "red": {"fov": "fov_0", "channel": "CD4"},
            "green": {"fov": "fov_0", "channel": "CD11c"},
        }
        composite_image = viewer_widget.create_composite_from_dataset(path_dict)
        assert isinstance(composite_image, np.ndarray)
        assert composite_image.shape == (256, 256, 3)


def test_overlay():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths, _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        path_dict = {
            "red": os.path.join(temp_dir, "fov_0", "CD4.tiff"),
            "green": os.path.join(temp_dir, "fov_0", "CD11c.tiff"),
        }
        # test if segmentation gets added
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))
        dataset = MultiplexDataset(fov_paths, naming_convention)
        viewer_widget = NimbusViewer(dataset, temp_dir)
        composite_image = viewer_widget.create_composite_image(path_dict)
        composite_image, seg_boundaries = viewer_widget.overlay(
            composite_image, add_boundaries=True
        )
        assert composite_image.shape == (256, 256, 3)
        assert seg_boundaries.shape == (256, 256)
        assert np.unique(seg_boundaries).tolist() == [0, 1]
