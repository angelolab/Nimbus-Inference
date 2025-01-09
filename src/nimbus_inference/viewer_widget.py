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
            val = (np.max(composite_image, axis=(0,1))*0.5).astype(composite_image.dtype)
            val = np.min(val[val>0])
            composite_image[seg_boundaries] = [val]*3
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

        # Initialize matplotlib figure
        with plt.ioff():
            self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
        
        # uncomment the following lines to enable zooming via scroll wheel
        # self.zoom_handler = self.custom_zoom_factory(self.ax[0])
        # self.pan_handler = panhandler(self.fig)
        
        # Display the figure canvas
        display(self.fig.canvas)

    def custom_zoom_factory(self, ax, base_scale=1.1):
        """Enable zooming via scroll wheel on matplotlib axes.

        Args:
            ax (matplotlib ax): ax to enable zooming on.
            base_scale (float): Scale factor for zooming.
        """
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw_idle()

        fig = ax.get_figure()  # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def update_left_image(self, image):
        """Update the left image displayed in the viewer.
        
        Args:
            image (np.array): Image to display.
        """
        self.ax[0].imshow(image)
        self.ax[0].title.set_text(self.title_left)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.fig.canvas.draw_idle()

    def update_right_image(self, image):
        """Update the right image displayed in the viewer.
        
        Args:
            image (np.array): Image to display.
        """
        self.ax[1].imshow(image, vmin=0, vmax=255)
        self.ax[1].title.set_text(self.title_right)
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
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
        self.overlay_checkbox = widgets.Checkbox(
            value=False,
            description='Overlay segmentations',
            disabled=False
        )


    def layout(self):
        """Creates layout for viewer."""
        channel_selectors = widgets.HBox([
            self.red_select,
            self.green_select,
            self.blue_select
        ])
        layout = widgets.HBox([
            # widgets.HBox([
                self.dataset_select,
                self.fov_select,
                channel_selectors,
                self.update_button
            # ]),
        ])
        display(layout)

    def select_dataset(self, change):
        """Selects dataset to display.

        Args:
            change (dict): Change dictionary from ipywidgets.
        """
        self.dataset = self.datasets.set_dataset(change['new'])
        self.fov_names = natsorted(copy(self.dataset.fovs))
        self.fov_select.options = self.fov_names
        self.select_fov(None)


    def update_img(self, image_fn, composite_image):
        """Updates image in viewer by saving it as png and loading it with the viewer widget.

        Args:
            ax (matplotlib ax): ax to update.
            composite_image (np.array): Composite image to display.
        """
        if composite_image.shape[0] > self.max_resolution[0] or composite_image.shape[1] > self.max_resolution[1]:
            scale = float(np.max(self.max_resolution)/np.max(composite_image.shape))
            composite_image = rescale(composite_image, (scale, scale, 1), preserve_range=True)
            composite_image = composite_image.astype(np.uint8)
        image_fn(composite_image)

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

        img = in_composite_image[...,0].astype(np.float32) * 0
        right_images = []
        for c, s in {'red': self.red_select.value,
                     'green': self.green_select.value,
                     'blue': self.blue_select.value}.items():
            if s:
                composite_image = self.dataset.get_groundtruth(
                    self.fov_select.value, s
                )
            else:
                composite_image = img
            composite_image = np.squeeze(composite_image).astype(np.float32)
            right_images.append(composite_image)
        right_images = np.stack(right_images, axis=-1)
        right_images = np.clip(right_images, 0, 2)
        right_images[right_images == 2] = 0.3
        right_images[seg_boundaries] = 0.0
        right_images *= 255.0
        right_images = right_images.astype(np.uint8)

        # update image viewers
        self.update_img(self.image.update_left_image, in_composite_image)
        self.update_img(self.image.update_right_image, right_images)
