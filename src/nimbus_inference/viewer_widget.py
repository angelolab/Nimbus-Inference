import os
import ipywidgets as widgets
from IPython.display import display
from io import BytesIO
from skimage import io
from copy import copy
import numpy as np
from natsort import natsorted
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from nimbus_inference.utils import MultiplexDataset, InteractiveDataset
from mpl_interactions import panhandler
import matplotlib.pyplot as plt

class NimbusViewer(object):
    """Viewer for Nimbus application.

    Args:
        dataset (MultiplexDataset): dataset object
        output_dir (str): Path to directory containing output of Nimbus application.
        segmentation_naming_convention (fn): Function that maps input path to segmentation path
        img_width (str): Width of images in viewer.
        suffix (str): Suffix of images in dataset.
        max_resolution (tuple): Maximum resolution of images in viewer.
    """
    def __init__(
            self, dataset: MultiplexDataset, output_dir: str, img_width='600px', suffix=".tiff",
            max_resolution=(2048, 2048)
        ):
        self.image_width = img_width
        self.dataset = dataset
        self.output_dir = output_dir
        self.suffix = suffix
        self.max_resolution = max_resolution
        self.fov_names = natsorted(copy(self.dataset.fovs))
        self.update_button = widgets.Button(description="Update Image")
        self.update_button.on_click(self.update_button_click)
        self.overlay_checkbox = widgets.Checkbox(
            value=True,
            description='Overlay segmentations',
            disabled=False
        )

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
        channels = natsorted(copy(self.dataset.channels))
        self.red_select.options = channels
        self.green_select.options = channels
        self.blue_select.options = channels

    def overlay(self, composite_image, add_boundaries=False, add_overlay=False):
        """Adds overlay to composite image.

        Args:
            composite_image (np.array): Composite image to add overlay to.
            boundaries (bool): Whether to add boundaries to overlay.
        Returns:
            np.array: Composite image with overlay.
        """
        seg_img = self.dataset.get_segmentation(self.fov_select.value)
        seg_boundaries = find_boundaries(seg_img, mode='inner')
        seg_img[seg_boundaries] = 0
        seg_img = np.clip(seg_img, 0, 1)
        seg_img = np.repeat(seg_img[..., np.newaxis], 3, axis=-1) * np.max(composite_image)
        background_mask = composite_image < np.max(composite_image) * 0.2

        if add_overlay:
            composite_image[background_mask] += (seg_img[background_mask] * 0.2).astype(
                composite_image.dtype
            )

        if add_boundaries:
            val = (np.max(composite_image, axis=(0, 1)) * 0.5).astype(composite_image.dtype)
            positive_val = val[val > 0]
            if positive_val.size > 0:
                val_min = np.min(positive_val)
                composite_image[seg_boundaries] = [val_min] * 3
            else:
                # Handle all-black image case (skip boundaries)
                pass
        else:
            seg_boundaries = None

        return composite_image, seg_boundaries


    def create_composite_from_dataset(self, path_dict):
        """Creates composite image from input paths.

        Args:
            path_dict (dict): Dictionary of paths to images.
        Returns:
            np.array: Composite image.
        """
        for k in ["red", "green", "blue"]:
            if k not in path_dict.keys():
                path_dict[k] = None
        output_image = []
        for p in list(path_dict.values()):
            if p:
                img = self.dataset.get_channel(p["fov"], p["channel"])
                output_image.append(img)
            else:
                p = [p for p in path_dict.values() if p][0]
                img = self.dataset.get_channel(p["fov"], p["channel"])
                output_image.append(img*0)
        composite_image = np.stack(output_image, axis=-1)
        return composite_image

    def create_composite_image(self, path_dict, add_overlay=True, add_boundaries=False):
        """Creates composite image from input paths.

        Args:
            path_dict (dict): Dictionary of paths to images.
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
        return composite_image

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
        viewer_html = widgets.HTML("<h2>Select files</h2>")
        input_html = widgets.HTML("<h2>Input</h2>")
        output_html = widgets.HTML("<h2>Nimbus Output</h2>")

        layout = widgets.HBox([
            widgets.VBox([
                viewer_html,
                self.fov_select,
                channel_selectors,
                self.overlay_checkbox,
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
            str: Path to similar filename.
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
        if composite_image.shape[0] > self.max_resolution[0] or composite_image.shape[1] > self.max_resolution[1]:
            scale = float(np.max(self.max_resolution)/np.max(composite_image.shape))
            composite_image = rescale(composite_image, (scale, scale, 1), preserve_range=True)
            composite_image = composite_image.astype(np.uint8)
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
                self.output_dir, self.fov_select.value, self.red_select.value + self.suffix
            )
            in_path_dict["red"] = {"fov": self.fov_select.value, "channel": self.red_select.value}
        if self.green_select.value:
            path_dict["green"] = os.path.join(
                self.output_dir, self.fov_select.value, self.green_select.value + self.suffix
            )
            in_path_dict["green"] = {
                "fov": self.fov_select.value, "channel": self.green_select.value
            }
        if self.blue_select.value:
            path_dict["blue"] = os.path.join(
                self.output_dir, self.fov_select.value, self.blue_select.value + self.suffix
            )
            in_path_dict["blue"] = {
                "fov": self.fov_select.value, "channel": self.blue_select.value
            }
        non_none = [p for p in path_dict.values() if p]
        if not non_none:
            return
        composite_image = self.create_composite_image(path_dict)
        composite_image, _ = self.overlay(
            composite_image, add_overlay=True
        )

        in_composite_image = self.create_composite_from_dataset(in_path_dict)
        in_composite_image, seg_boundaries = self.overlay(
            in_composite_image, add_boundaries=self.overlay_checkbox.value
        )
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


class InteractiveImageDuo(widgets.Image):
    """Interactive image viewer for Nimbus application.

    Args:
        figsize (tuple): Size of figure.
        title_left (str): Title of left image.
        title_right (str): Title of right image.
    """
    def __init__(self, figsize=(10, 5), title_left='Multiplexed image', title_right='Groundtruth'):
        super().__init__()
        self.title_left = title_left
        self.title_right = title_right

        # Initialize matplotlib figure and image objects
        with plt.ioff():
            self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
            self.ax[0].set_title(self.title_left)
            self.ax[1].set_title(self.title_right)
            self.ax[0].set_xticks([])
            self.ax[0].set_yticks([])
            self.ax[1].set_xticks([])
            self.ax[1].set_yticks([])
        
        self.left_image = None
        self.right_image = None
        
        # Display the figure canvas
        display(self.fig.canvas)

    def update_left_image(self, image):
        """Update the left image displayed in the viewer.
        
        Args:
            image (np.array): Image to display.
        """
        if self.left_image is None:
            self.left_image = self.ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            self.left_image.set_data(image)
        self.fig.canvas.draw_idle()

    def update_right_image(self, image):
        """Update the right image displayed in the viewer.
        
        Args:
            image (np.array): Image to display.
        """
        if self.right_image is None:
            self.right_image = self.ax[1].imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            self.right_image.set_data(image)
        self.fig.canvas.draw_idle()


class NimbusInteractiveGTViewer(NimbusViewer):
    """Interactive viewer for Nimbus application that shows input data and ground truth
    side by side.

    Args:
        dataset (MultiplexDataset): dataset object
        output_dir (str): Path to directory containing output of Nimbus application.
        figsize (tuple): Size of figure.
    """
    def __init__(
            self, datasets: InteractiveDataset, output_dir, figsize=(20, 10)
        ):
        super().__init__(
            datasets.datasets[datasets.dataset_names[0]], output_dir
        )
        self.image = InteractiveImageDuo(figsize=figsize)
        self.dataset = datasets.datasets[datasets.dataset_names[0]]
        self.datasets = datasets
        self.dataset_select = widgets.Select(
            options=datasets.dataset_names,
            description='Dataset:',
            disabled=False
        )
        self.dataset_select.observe(self.select_dataset, names='value')

        # Replace three channel selectors with a single one
        self.channel_select = widgets.Select(
            options=natsorted(copy(self.dataset.channels)),
            description='Channel:',
            disabled=False
        )

        # Add intensity sliders
        self.min_intensity_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.01, description="Min Intensity"
        )
        self.max_intensity_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=1.0, step=0.01, description="Max Intensity"
        )
        self.min_intensity_slider.observe(self.update_intensity, names='value')
        self.max_intensity_slider.observe(self.update_intensity, names='value')

    def layout(self):
        """Creates layout for viewer."""
        sliders = widgets.VBox([
            self.min_intensity_slider,
            self.max_intensity_slider
        ])
        layout = widgets.HBox([
            # Widgets layout
            self.dataset_select,
            self.fov_select,
            self.channel_select,
            sliders,
            self.overlay_checkbox,
            self.update_button
        ])
        display(layout)

    def select_dataset(self, change):
        """Selects dataset and resets sliders/channels to defaults."""
        self.dataset = self.datasets.set_dataset(change['new'])
        self.fov_names = natsorted(copy(self.dataset.fovs))
        self.fov_select.options = self.fov_names
        
        # Reset channel selection to avoid stale references
        self.channel_select.options = natsorted(copy(self.dataset.channels))
        self.channel_select.value = None  # Clear current selection
        
        # Reset sliders
        self.min_intensity_slider.value = 0.0
        self.max_intensity_slider.value = 1.0
        self.select_fov(None)

    def update_intensity(self, change=None):
        """Update the left image intensity scaling based on the slider values."""
        self.update_composite()

    def select_fov(self, change):
        """Selects FOV and resets sliders to defaults."""
        super().select_fov(change)  # Call parent method
        # Reset sliders when FOV changes
        self.min_intensity_slider.value = 0.0
        self.max_intensity_slider.value = 1.0

    def select_dataset(self, change):
        """Selects dataset and resets sliders to defaults."""
        self.dataset = self.datasets.set_dataset(change['new'])
        self.fov_names = natsorted(copy(self.dataset.fovs))
        self.fov_select.options = self.fov_names
        self.channel_select.options = natsorted(copy(self.dataset.channels))
        self.select_fov(None)
        # Reset sliders when dataset changes
        self.min_intensity_slider.value = 0.0
        self.max_intensity_slider.value = 1.0

    def overlay(self, composite_image, add_boundaries=False, add_overlay=False):
        """Adds overlay to composite image."""
        seg_img = self.dataset.get_segmentation(self.fov_select.value)
        seg_boundaries = find_boundaries(seg_img, mode='inner')
        seg_img[seg_boundaries] = 0
        seg_img = np.clip(seg_img, 0, 1)
        seg_img = np.repeat(seg_img[..., np.newaxis], 3, axis=-1) * np.max(composite_image)
        background_mask = composite_image < np.max(composite_image) * 0.2

        if add_overlay:
            composite_image[background_mask] += (seg_img[background_mask] * 0.2).astype(
                composite_image.dtype
            )

        if add_boundaries:
            # Set boundaries to red [R, G, B] = [255, 0, 0]
            composite_image[seg_boundaries] = [255, 0, 0]
        else:
            seg_boundaries = None

        return composite_image, seg_boundaries

    def update_composite(self):
        """Updates composite image in viewer."""
        if not self.channel_select.value or self.channel_select.value not in self.dataset.channels:
            # Clear images if channel is invalid
            self.image.update_left_image(np.zeros((100, 100, 3), dtype=np.uint8))  # Black placeholder
            self.image.update_right_image(np.zeros((100, 100), dtype=np.uint8))
            return

        selected_channel = self.channel_select.value
        selected_fov = self.fov_select.value

        # Get raw image and normalize using 99th percentile
        raw_image = self.dataset.get_channel(selected_fov, selected_channel)
        img_min = raw_image.min()
        img_max = np.percentile(raw_image, 99)  # 99th percentile

        if img_max > img_min:
            normalized_image = (raw_image - img_min) / (img_max - img_min)
        else:
            normalized_image = raw_image

        # Apply intensity scaling
        min_intensity = self.min_intensity_slider.value
        max_intensity = self.max_intensity_slider.value
        if max_intensity <= min_intensity:
            max_intensity = min_intensity + 1e-6

        scaled_image = np.clip(
            (normalized_image - min_intensity) / (max_intensity - min_intensity),
            0.0, 1.0
        )
        scaled_image = (scaled_image * 255).astype(np.uint8)

        # Overlay boundaries if enabled (keep as RGB)
        scaled_image_rgb = np.repeat(scaled_image[..., np.newaxis], 3, axis=-1)
        composite_image, seg_boundaries = self.overlay(
            scaled_image_rgb, 
            add_boundaries=self.overlay_checkbox.value,
            add_overlay=False  # Explicitly set if needed
        )

        # Update left image with RGB composite (no grayscale conversion!)
        self.image.update_left_image(composite_image)

        # Update right image (ground truth)
        right_image = self.dataset.get_groundtruth(selected_fov, selected_channel)
        right_image = np.clip(right_image, 0, 2).astype(np.float32)
        right_image[right_image == 2] = 0.3
        right_image = np.squeeze(right_image)

        if seg_boundaries is None:
            seg_img = self.dataset.get_segmentation(self.fov_select.value)
            seg_boundaries = find_boundaries(seg_img, mode='inner')
        right_image[seg_boundaries] = 0.0

        right_image = (right_image * 255).astype(np.uint8)
        self.image.update_right_image(right_image)