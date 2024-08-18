import os
import ipywidgets as widgets
from IPython.core.display import HTML
from IPython.display import display
from io import BytesIO
from skimage import io
import numpy as np
from natsort import natsorted
from skimage.segmentation import find_boundaries


class CellAnalyzer(object):
    """Cell Analyzer for Nimbus application, can be used to compare and annotate cell images.

    Args:
        input_dir (str): Path to directory containing individual channels of multiplexed images
        cell_df (pd.DataFrame): DataFrame with columns 'pixie_ct', 'nimbus_ct', 'fov', 'label'
        output_dir (str): Path to save output annotations
        segmentation_naming_convention (fn): Function that maps input path to segmentation path
        img_width (str): Width of images in viewer.
        context (int): area around the cell to display
    """
    def __init__(
            self, input_dir, cell_df, output_dir,
            segmentation_naming_convention, img_width='1000px', context=None
        ):
        self.image_width = img_width
        self.input_dir = input_dir
        self.segmentation_naming_convention = segmentation_naming_convention
        self.cell_df = cell_df
        if "annotations" not in self.cell_df.columns:
            self.cell_df["annotations"] = [None]*len(self.cell_df)
        self.cell_ids = self.cell_df["Cell ID"].values.tolist()
        self.fov_names = self.cell_df["fov"].values
        self.output_dir = output_dir
        
        self.context = context
        display(HTML(
            "<style>.bottom_spacing_class {margin-bottom: 10px;}</style>"
        ))
        display(HTML(
            "<style>.left_spacing_class {margin-left: 60px;}</style>"
        ))
        display(HTML("<style>.font_size_class {line-height:30px;font-weight:500;font-size:200%;\
                     margin-left: 30px;}</style>"))
        self.update_button = widgets.Button(description="Update Image")
        self.update_button.on_click(self.update_button_click)
        self.update_button.add_class(
              "left_spacing_class"            
        )
        self.overlay_checkbox = widgets.Checkbox(
            value=True,
            description='Overlay segmentations',
            disabled=False
        )
        self.model_button = widgets.RadioButtons(
            options=["Nimbus", "Pixie", "None"], description="Cell type",
            style=dict(font_size="18px")
        )
        self.model_button.add_class("font_size_class")
        
        self.save_annotations_button = widgets.Button(description="Save and next cell")
        self.save_annotations_button.on_click(self.save_annotations_button_click)
        self.save_annotations_button.add_class("left_spacing_class")

        self.cell_id_select = widgets.Select(
            options=self.cell_ids,
            description='Cell ID:',
            disabled=False
        )
        self.cell_id_select.observe(self.cell_id_select_fn, names='value')
        
        self.red_select = widgets.Select(
            options=[],
            description='Red:',
            disabled=False,
            layout=widgets.Layout(height='120px')
        )
        self.green_select = widgets.Select(
            options=[],
            description='Green:',
            disabled=False,
            layout=widgets.Layout(height='120px')
        )
        self.blue_select = widgets.Select(
            options=[],
            description='Blue:',
            disabled=False,
            layout=widgets.Layout(height='120px')
        )
        self.input_image = widgets.Image()
        self.output_image = widgets.Image()

    def cell_id_select_fn(self, change):
        """Selects fov to display.

        Args:
            change (dict): Change dictionary from ipywidgets.
        """
        fov = self.cell_df[self.cell_df["Cell ID"] == self.cell_id_select.value]["fov"].values[0]
        fov_path = os.path.join(self.input_dir, fov)
        channels = [
            ch for ch in os.listdir(fov_path) if os.path.isfile(os.path.join(fov_path, ch))
        ]
        channels = [ch.split(".")[0] for ch in channels]
        self.red_select.options = natsorted(channels)
        self.green_select.options = natsorted(channels)
        self.blue_select.options = natsorted(channels)
        nimbus_ct = self.cell_df[
            self.cell_df["Cell ID"] == self.cell_id_select.value
        ]["nimbus_ct"].values[0]
        pixie_ct = self.cell_df[
            self.cell_df["Cell ID"] == self.cell_id_select.value
        ]["pixie_ct"].values[0]
        self.model_button.options = [nimbus_ct, pixie_ct, "None of the above"]

    def create_composite_image(self, path_dict, add_boundaries=True):
        """Creates composite image from input paths.

        Args:
            path_dict (dict): Dictionary of paths to images.
            add_boundaries (bool): Whether to add boundaries to multiplex image.
        Returns:
            np.array: Composite image.
        """
        for k in ["red", "green", "blue"]:
            if k not in path_dict.keys():
                path_dict[k] = None
        output_image = []
        for k, p in path_dict.items():
            if p:
                img = io.imread(p)
                output_image.append(img)
            else:
                non_none = [p for p in path_dict.values() if p]
                img = io.imread(non_none[0])
                output_image.append(img*0)
        # add overlay of instances
        composite_image = np.stack(output_image, axis=-1)
        if self.segmentation_naming_convention and add_boundaries:
            fov_path = os.path.split(list(path_dict.values())[0])[0]
            seg_path = self.segmentation_naming_convention(fov_path)
            seg_img = io.imread(seg_path)
            seg_boundaries = find_boundaries(seg_img, mode='inner')
            individual_cell_boundary = seg_boundaries.copy()
            cell_mask = seg_img == self.cell_df[
                self.cell_df["Cell ID"] == self.cell_id_select.value
            ]["label"].values[0]
            individual_cell_boundary = np.logical_and(cell_mask, seg_boundaries)
            val = (np.max(composite_image, axis=(0,1))*0.5).astype(composite_image.dtype)
            val = np.min(val[val>0])
        else:
            seg_boundaries = None
            individual_cell_boundary = None
        return composite_image, seg_boundaries, individual_cell_boundary
    
    def create_instance_image(self, path_dict, cell_id):
        """Creates composite image from input paths.

        Args:
            path_dict (dict): Dictionary of paths to images.
            cell_id (int): id of cell to highlight
        Returns:
            np.array: Composite image.
        """
        # add overlay of instances
        fov_path = os.path.split(list(path_dict.values())[0])[0]
        seg_path = self.segmentation_naming_convention(fov_path)
        seg_img = io.imread(seg_path)
        seg_boundaries = find_boundaries(seg_img, mode='inner')
        seg_img[seg_boundaries] = 0
        seg_img_clipped = np.clip(seg_img, 0, 1) * 0.2
        seg_img_clipped[seg_img == cell_id] = 1
        return seg_img_clipped

    def layout(self):
        """Creates layout for viewer."""
        channel_selectors = widgets.VBox([
            self.red_select,
            self.green_select,
            self.blue_select
        ])
        self.input_image.layout.width = self.image_width
        self.output_image.layout.width = self.image_width
        self.input_image.layout.height = self.image_width
        self.output_image.layout.height = self.image_width

        layout = widgets.HBox([
            widgets.VBox([
                self.cell_id_select,
                channel_selectors,
                self.overlay_checkbox,
                self.update_button,
            widgets.VBox([
                self.model_button,
                self.save_annotations_button
            ]),
            ], layout=widgets.Layout(width='30%')),
        widgets.HBox([
            # input_html,
            self.input_image
        ]),
        widgets.HBox([
            # output_html,
            self.output_image
        ])
        ])
        display(layout)

    def search_for_similar(self, select_value):
        """Searches for similar filename in input directory.

        Args:
            select_value (str): Filename to search for.
        Returns:
            str: Path to similar filename.
        """
        fov = self.cell_df[self.cell_df["Cell ID"] == self.cell_id_select.value]["fov"].values[0]
        in_f_path = os.path.join(self.input_dir, fov)
        # search for similar filename in in_f_path
        in_f_files = [
            f for f in os.listdir(in_f_path) if os.path.isfile(os.path.join(in_f_path, f))
        ]
        similar_path = None
        for f in in_f_files:
            if select_value + "." in f:
                similar_path = os.path.join(self.input_dir, fov, f)
        return similar_path

    def update_img(self, image_viewer, composite_image):
        """Updates image in viewer by saving it as png and loading it with the viewer widget.

        Args:
            image_viewer (ipywidgets.Image): Image widget to update.
            composite_image (np.array): Composite image to display.
        """
        # Convert composite image to bytes and assign it to the output_image widget
        with BytesIO() as output_buffer:
            io.imsave(output_buffer, composite_image, format="png")
            output_buffer.seek(0)
            image_viewer.value = output_buffer.read()

    def update_composite(self):
        """Updates composite image in viewer."""
        in_path_dict = {
            "red": None,
            "green": None,
            "blue": None
        }
        if self.red_select.value:
            in_path_dict["red"] = self.search_for_similar(self.red_select.value)
        if self.green_select.value:
            in_path_dict["green"] = self.search_for_similar(self.green_select.value)
        if self.blue_select.value:
            in_path_dict["blue"] = self.search_for_similar(self.blue_select.value)
        non_none = [p for p in in_path_dict.values() if p]
        if not non_none:
            return
        in_composite_image, seg_boundaries, individual_cell_boundary = self.create_composite_image(
            in_path_dict, add_boundaries=self.overlay_checkbox.value
        )
        in_composite_image = in_composite_image / np.quantile(
            in_composite_image, 0.999, axis=(0,1)
        )
        in_composite_image = np.clip(in_composite_image*255, 0, 255).astype(np.uint8)
        if seg_boundaries is not None:
            in_composite_image[seg_boundaries] = [75, 75, 75]
            in_composite_image[individual_cell_boundary] = [175, 175, 175]
        cell_label = self.cell_df[
            self.cell_df["Cell ID"] == self.cell_id_select.value
        ]["label"].values[0]
        seg_image = self.create_instance_image(in_path_dict, cell_label)
        seg_image = (seg_image * 255).astype(np.uint8)
        if self.context:
            h, w = np.where(seg_image == seg_image.max())
            h, w = np.mean(h).astype(np.uint16), np.mean(w).astype(np.uint16)
            h_max, w_max = seg_image.shape
            h_0, h_1 = np.clip(h-self.context, 0, h_max), np.clip(h+self.context, 0, h_max)
            w_0, w_1 = np.clip(w-self.context, 0, w_max), np.clip(w+self.context, 0, w_max)
            seg_image = seg_image[h_0:h_1, w_0:w_1]
            in_composite_image = in_composite_image[h_0:h_1, w_0:w_1]
        # update image viewers
        self.update_img(self.input_image, in_composite_image)
        self.update_img(self.output_image, seg_image)

    def update_button_click(self, button):
        """Updates composite image in viewer when update button is clicked.
        
        Args:
            button (ipywidgets.Button): Button widget that was clicked.
        """
        self.update_composite()
    
    def save_annotations_button_click(self, button):
        """Updates composite image in viewer when pixie button is clicked.
        
        Args:
            button (ipywidgets.Button): Button widget that was clicked.
        """
        self.cell_df.loc[
            self.cell_df["Cell ID"] == self.cell_id_select.value, "annotations"
        ] = self.model_button.value
        self.cell_df.to_csv(
            os.path.join(self.output_dir, "annotations.csv"), index=False
        )
        # bump to next cell
        current_index = self.cell_id_select.options.index(self.cell_id_select.value)
        max_index = len(self.cell_id_select.options)
        if current_index + 1 < max_index:
            self.cell_id_select.value = self.cell_id_select.options[current_index + 1]
        self.update_composite()

    def display(self):
        """Displays viewer."""
        self.cell_id_select_fn(None)
        self.layout()
        self.update_composite() 