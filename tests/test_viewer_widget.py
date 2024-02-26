from nimbus_inference.viewer_widget import NimbusViewer
from tests.test_utils import prepare_ome_tif_data, prepare_tif_data
import numpy as np
import tempfile
import os


def test_NimbusViewer():
    with tempfile.TemporaryDirectory() as temp_dir:
        _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        viewer_widget = NimbusViewer(temp_dir, temp_dir)
        assert isinstance(viewer_widget, NimbusViewer)


def test_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        viewer_widget = NimbusViewer(temp_dir, temp_dir)
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