from nimbus_inference.cell_analyzer import CellAnalyzer
from nimbus_inference.nimbus import Nimbus, prep_naming_convention
from tests.test_utils import prepare_tif_data
import pandas as pd
import numpy as np
import tempfile
import os

cell_df = pd.DataFrame(
    {
     "Cell ID": [1, 2, 3, 4],
     "pixie_ct": ["Eosinophil", "Neutrophil", "Lymphocyte", "Monocyte"],
     "nimbus_ct": ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil"],
     "fov": ["fov_0", "fov_0", "fov_1", "fov_1"],
     "label": [1, 2, 3, 4],
     } 
)

def test_CellAnalyzer():
    with tempfile.TemporaryDirectory() as temp_dir:
        _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))

        viewer_widget = CellAnalyzer(
            temp_dir, cell_df=cell_df, segmentation_naming_convention=naming_convention,
            output_dir=temp_dir
        )
        assert isinstance(viewer_widget, CellAnalyzer)


def test_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        _ = prepare_tif_data(
            num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD11c", "CD56"]
        )
        naming_convention = prep_naming_convention(os.path.join(temp_dir, "deepcell_output"))
        viewer_widget = CellAnalyzer(
            temp_dir, cell_df=cell_df, segmentation_naming_convention=naming_convention,
            output_dir=temp_dir
        )
        path_dict = {
            "red": os.path.join(temp_dir, "fov_0", "CD4.tiff"),
            "green": os.path.join(temp_dir, "fov_0", "CD11c.tiff"),
        }
        composite_image, _, _ = viewer_widget.create_composite_image(path_dict)
        assert isinstance(composite_image, np.ndarray)
        assert composite_image.shape == (256, 256, 3)

        path_dict["blue"] = os.path.join(temp_dir, "fov_0", "CD56.tiff")
        composite_image, _, _ = viewer_widget.create_composite_image(path_dict)
        assert composite_image.shape == (256, 256, 3)
        # test if segmentation gets added
        viewer_widget = CellAnalyzer(
            temp_dir, cell_df=cell_df, segmentation_naming_convention=naming_convention,
            output_dir=temp_dir
        )
        composite_image, seg_boundaries, cell_boundaries = viewer_widget.create_composite_image(
            path_dict
        )
        assert composite_image.shape == (256, 256, 3)
        assert seg_boundaries.shape == (256, 256)
        assert cell_boundaries.shape == (256, 256)
        assert np.unique(seg_boundaries).tolist() == [0, 1]
