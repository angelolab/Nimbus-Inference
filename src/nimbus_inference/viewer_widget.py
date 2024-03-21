import os
import ipywidgets as widgets
from IPython.display import display
from io import BytesIO
from skimage import io
from copy import copy
import numpy as np
from natsort import natsorted
from skimage.segmentation import find_boundaries


class NimbusViewer(object):
    def __init__(self, input_dir, output_dir, segmentation_naming_convention=None, img_width='600px'):
        """Viewer for Nimbus application.
        Args:
            input_dir (str): Path to directory containing individual channels of multiplexed images
            output_dir (str): Path to directory containing output of Nimbus application.
            segmentation_naming_convention (fn): Function that maps input path to segmentation path
            img_width (str): Width of images in viewer.
        """
        self.image_width = img_width
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.segmentation_naming_convention = segmentation_naming_convention
        self.fov_names = [os.path.basename(p) for p in os.listdir(output_dir) if \
                          os.path.isdir(os.path.join(output_dir, p))]
        self.fov_names = natsorted(self.fov_names)
        self.update_button = widgets.Button(description="Update Image")
        self.update_button.on_click(self.update_button_click)

        self.fov_select = widgets.Select(
            options=self.fov_names,
            description='FOV:',
            disabled=False
        )
        self.fov_select.observe(self.select_fov, names='value')

        self.red_select = widgets.Select(
            options=[],
            description='Red:',
            disabled=False
        )
        self.green_select = widgets.Select(
            options=[],
            description='Green:',
            disabled=False
        )
        self.blue_select = widgets.Select(
            options=[],
            description='Blue:',
            disabled=False
        )
        self.input_image = widgets.Image()
        self.output_image = widgets.Image()

    def select_fov(self, change):
        """Selects fov to display.
        Args:
            change (dict): Change dictionary from ipywidgets.
        """
        fov_path = os.path.join(self.output_dir, self.fov_select.value)
        channels = [
            ch for ch in os.listdir(fov_path) if os.path.isfile(os.path.join(fov_path, ch))
        ]
        self.red_select.options = natsorted(channels)
        self.green_select.options = natsorted(channels)
        self.blue_select.options = natsorted(channels)

    def create_composite_image(self, path_dict, add_overlay=True, add_boundaries=False):
        """Creates composite image from input paths.
        Args:
            path_dict (dict): Dictionary of paths to images.
        Returns:
            composite_image (np.array): Composite image.
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
        if self.segmentation_naming_convention and add_overlay:
            fov_path = os.path.split(list(path_dict.values())[0])[0]
            seg_path = self.segmentation_naming_convention(fov_path)
            seg_img = io.imread(seg_path)
            seg_boundaries = find_boundaries(seg_img, mode='inner')
            seg_img[seg_boundaries] = 0
            seg_img = np.clip(seg_img, 0, 1)
            seg_img = np.repeat(seg_img[..., np.newaxis], 3, axis=-1) * np.max(composite_image)
            background_mask = composite_image < np.max(composite_image) * 0.2
            composite_image[background_mask] += (seg_img[background_mask] * 0.2).astype(composite_image.dtype)
        elif self.segmentation_naming_convention and add_boundaries:
            fov_path = os.path.split(list(path_dict.values())[0])[0]
            seg_path = self.segmentation_naming_convention(fov_path)
            seg_img = io.imread(seg_path)
            seg_boundaries = find_boundaries(seg_img, mode='inner')
            val = (np.max(composite_image, axis=(0,1))*0.5).astype(composite_image.dtype)
            val = np.min(val[val>0])
            composite_image[seg_boundaries] = [val]*3 
        else:
            seg_boundaries = None
        return composite_image, seg_boundaries

    def layout(self):
        """Creates layout for viewer."""
        channel_selectors = widgets.VBox([
            self.red_select,
            self.green_select,
            self.blue_select
        ])
        self.input_image.layout.width = self.image_width
        self.output_image.layout.width = self.image_width
        viewer_html = widgets.HTML("<h2>Select files</h2>")
        input_html = widgets.HTML("<h2>Input</h2>")
        output_html = widgets.HTML("<h2>Nimbus Output</h2>")

        layout = widgets.HBox([
            widgets.VBox([
                viewer_html,
                self.fov_select,
                channel_selectors,
                self.update_button
            ]),
        widgets.VBox([
            input_html,
            self.input_image
        ]),
        widgets.VBox([
            output_html,
            self.output_image
        ])
        ])
        display(layout)

    def search_for_similar(self, select_value):
        """Searches for similar filename in input directory.
        Args:
            select_value (str): Filename to search for.
        Returns:
            similar_path (str): Path to similar filename.
        """
        in_f_path = os.path.join(self.input_dir, self.fov_select.value)
        # search for similar filename in in_f_path
        in_f_files = [
            f for f in os.listdir(in_f_path) if os.path.isfile(os.path.join(in_f_path, f))
        ]
        similar_path = None
        for f in in_f_files:
            if select_value.split(".")[0]+"." in f:
                similar_path = os.path.join(self.input_dir, self.fov_select.value, f)
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
        path_dict = {
            "red": None,
            "green": None,
            "blue": None
        }
        in_path_dict = copy(path_dict)
        if self.red_select.value:
            path_dict["red"] = os.path.join(
                self.output_dir, self.fov_select.value, self.red_select.value
            )
            in_path_dict["red"] = self.search_for_similar(self.red_select.value)
        if self.green_select.value:
            path_dict["green"] = os.path.join(
                self.output_dir, self.fov_select.value, self.green_select.value
            )
            in_path_dict["green"] = self.search_for_similar(self.green_select.value)
        if self.blue_select.value:
            path_dict["blue"] = os.path.join(
                self.output_dir, self.fov_select.value, self.blue_select.value
            )
            in_path_dict["blue"] = self.search_for_similar(self.blue_select.value)
        non_none = [p for p in path_dict.values() if p]
        if not non_none:
            return
        composite_image, _ = self.create_composite_image(path_dict)
        in_composite_image, seg_boundaries = self.create_composite_image(in_path_dict, add_overlay=False, add_boundaries=True)
        in_composite_image = in_composite_image / np.quantile(
            in_composite_image, 0.999, axis=(0,1)
        )
        in_composite_image = np.clip(in_composite_image*255, 0, 255).astype(np.uint8)
        if seg_boundaries is not None:
            in_composite_image[seg_boundaries] = [127, 127, 127]
        # update image viewers
        self.update_img(self.input_image, in_composite_image)
        self.update_img(self.output_image, composite_image)

    def update_button_click(self, button):
        """Updates composite image in viewer when update button is clicked."""
        self.update_composite()
    
    def display(self):
        """Displays viewer."""
        self.select_fov(None)
        self.layout()
        self.update_composite() 