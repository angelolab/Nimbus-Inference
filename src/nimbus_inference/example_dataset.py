# Code copied from github.com/angelolab/ark-analysis
import pathlib
import shutil
import warnings
from typing import Union
import datasets
from alpineer.misc_utils import verify_in_list
import zipfile
import os
import requests

EXAMPLE_DATASET_REVISION: str = "main"


class ExampleDataset:
    def __init__(self, dataset: str, overwrite_existing: bool = True, cache_dir: str = None,
                 revision: str = None) -> None:
        """
        Constructs a utility class for downloading and moving the dataset with respect to it's
        various partitions on Hugging Face: https://huggingface.co/datasets/angelolab/ark_example.

        Args:
            dataset (str): The name of the dataset to download. Can be one of

                    * `"segment_image_data"`
                    * `"cluster_pixels"`
                    * `"cluster_cells"`
                    * `"post_clustering"`
                    * `"fiber_segmentation"`
                    * `"LDA_preprocessing"`
                    * `"LDA_training_inference"`
                    * `"neighborhood_analysis"`
                    * `"pairwise_spatial_enrichment"`
                    * `"ome_tiff"`
                    * `"ez_seg_data"`
            overwrite_existing (bool): A flag to overwrite existing data. Defaults to `True`.
            cache_dir (str, optional): The directory to save the cache dir. Defaults to `None`,
                which internally in Hugging Face defaults to `~/.cache/huggingface/datasets`.
            revision (str, optional): The commit ID from Hugging Face for the dataset. Used for
                internal development only. Allows the user to fetch a commit from a particular
                `revision` (Hugging Face's terminology for branch). Defaults to `None`. Which
                defaults to the latest version in the `main` branch.
                (https://huggingface.co/datasets/angelolab/ark_example/tree/main).
        """
        self.dataset_paths = None
        self.dataset = dataset
        self.overwrite_existing = overwrite_existing
        self.cache_dir = cache_dir if cache_dir else pathlib.Path("~/.cache/huggingface/datasets").expanduser()
        self.revision = revision

        self.path_suffixes = {
            "image_data": "image_data",
            "cell_table": "segmentation/cell_table",
            "deepcell_output": "segmentation/deepcell_output",
            "example_pixel_output_dir": "pixie/example_pixel_output_dir",
            "example_cell_output_dir": "pixie/example_cell_output_dir",
            "spatial_lda": "spatial_analysis/spatial_lda",
            "post_clustering": "post_clustering",
            "ome_tiff": "ome_tiff",
            "ez_seg_data": "ez_seg_data"
        }
        """
        Path suffixes for mapping each downloaded dataset partition to it's appropriate
        relative save directory.
        """

    def download_example_dataset(self):
        """
        Downloads the example dataset from Hugging Face Hub.
        The following is a link to the dataset used:
        https://huggingface.co/datasets/angelolab/ark_example

        The dataset will be downloaded to the Hugging Face default cache
        `~/.cache/huggingface/datasets`.
        """
        ds_paths = datasets.load_dataset(path="angelolab/ark_example",
                                                   revision=self.revision,
                                                   name=self.dataset,
                                                   cache_dir=self.cache_dir,
                                                   token=False,
                                                   trust_remote_code=True)

        # modify the paths to be relative to the os
        # For example:
        # '/Users/user/.cache/huggingface/datasets/downloads/extracted/<hash>'
        # becomes 'pathlib.path(self.dataset_cache) / downloads/extracted/<hash>/<feature_name>'
        self.dataset_paths = {}
        for ds_name,ds in ds_paths.items():
            self.dataset_paths[ds_name] = {}
            for feature in ds.features:
                p, = ds[feature]
                # extract the path relative to the cache_dir (last 3 parts of the path)
                p = pathlib.Path(*pathlib.Path(p).parts[-3:])
                # Set the start of the path to the cache_dir (for the user's machine)
                self.dataset_paths[ds_name][feature] = self.cache_dir / p / feature


    def check_empty_dst(self, dst_path: pathlib.Path) -> bool:
        """
        Checks to see if the folder for a dataset config already exists in the `save_dir`
        (i.e. `dst_path` is the specific folder for the config.). If the folder exists, and
        there are no contents, then it'll return True, False otherwise.

        Args:
            dst_path (pathlib.Path): The destination directory to check to see if
            files exist in it.

        Returns:
            bool: Returns `True` if there are no files in the directory `dst_path`.
                Returns `False` if there are files in that directory `dst_path`.
        """
        dst_files = list(dst_path.rglob("*"))

        if len(dst_files) == 0:
            return True
        else:
            return False

    def move_example_dataset(self, move_dir: Union[str, pathlib.Path]):
        """
        Moves the downloaded example data from the `cache_dir` to the `save_dir`.

        Args:
            move_dir (Union[str, pathlib.Path]): The path to save the dataset files in.
        """
        if type(move_dir) is not pathlib.Path:
            move_dir = pathlib.Path(move_dir)

        dataset_names = list(self.dataset_paths[self.dataset].keys())

        for ds_n in dataset_names:
            ds_n_suffix: str = pathlib.Path(self.path_suffixes[ds_n])

            # The path where the dataset is saved in the Hugging Face Cache post-download,
            # Necessary to copy + move the data from the cache to the user specified `move_dir`.
            src_path = pathlib.Path(self.dataset_paths[self.dataset][ds_n])
            dst_path: pathlib.Path = move_dir / ds_n_suffix

            # Overwrite the existing dataset when `overwrite_existing` == `True`
            # and when the `dst_path` is empty.

            # `True` if `dst_path` is empty, `False` if data exists in `dst_path`
            empty_dst_path: bool = self.check_empty_dst(dst_path=dst_path)

            if self.overwrite_existing:
                if not empty_dst_path:
                    warnings.warn(UserWarning(f"Files exist in {dst_path}. \
                        They will be overwritten by the downloaded example dataset."))

                # Remove files in the destination path
                [f.unlink() for f in dst_path.glob("*") if f.is_file()]
                # Fill destination path
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(r"\.\!*"))
            else:
                if empty_dst_path:
                    warnings.warn(UserWarning(f"Files do not exist in {dst_path}. \
                        The example dataset will be added in."))
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns(r"\.\!*"))
                else:
                    warnings.warn(UserWarning(f"Files exist in {dst_path}. \
                        They will not be overwritten."))


def get_example_dataset(dataset: str, save_dir: Union[str, pathlib.Path],
                        overwrite_existing: bool = True):
    """
    A user facing wrapper function which downloads a specified dataset from Hugging Face,
    and moves it to the specified save directory `save_dir`.
    The dataset may be found here: https://huggingface.co/datasets/angelolab/ark_example


    Args:
        dataset (str): The name of the dataset to download. Can be one of

                * `"segment_image_data"`
                * `"cluster_pixels"`
                * `"cluster_cells"`
                * `"post_clustering"`
                * `"fiber_segmentation"`
                * `"LDA_preprocessing"`
                * `"LDA_training_inference"`
                * `"neighborhood_analysis"`
                * `"pairwise_spatial_enrichment"`
                * `"ez_seg_data"`
        save_dir (Union[str, pathlib.Path]): The path to save the dataset files in.
        overwrite_existing (bool): The option to overwrite existing configs of the `dataset`
            downloaded. Defaults to True.
    """

    valid_datasets = ["segment_image_data",
                      "cluster_pixels",
                      "cluster_cells",
                      "post_clustering",
                      "fiber_segmentation",
                      "LDA_preprocessing",
                      "LDA_training_inference",
                      "neighborhood_analysis",
                      "pairwise_spatial_enrichment",
                      "ome_tiff",
                      "ez_seg_data"]

    # Check the appropriate dataset name
    try:
        verify_in_list(dataset=dataset, valid_datasets=valid_datasets)
    except ValueError:
        err_str: str = f"""The dataset \"{dataset}\" is not one of the valid datasets available.
        The following are available: {*valid_datasets,}"""
        raise ValueError(err_str) from None

    example_dataset = ExampleDataset(dataset=dataset, overwrite_existing=overwrite_existing,
                                     cache_dir=None,
                                     revision=EXAMPLE_DATASET_REVISION)

    # Download the dataset
    example_dataset.download_example_dataset()

    # Move the dataset over to the save_dir from the user.
    example_dataset.move_example_dataset(move_dir=save_dir)


def download_and_unpack_gold_standard(save_dir: Union[str, pathlib.Path], overwrite_existing: bool = True):
    """
    Downloads 'gold_standard_labelled.zip' from the Hugging Face dataset and unpacks it in the given folder
    if the dataset is not already present there.

    Args:
        save_dir (Union[str, Path]): The path to save the dataset files in.
        overwrite_existing (bool): The option to overwrite existing files. Defaults to True.
    """
    url = "https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard/resolve/main/gold_standard_labelled.zip"
    save_dir = pathlib.Path(save_dir)
    zip_path = save_dir / "gold_standard_labelled.zip"

    # Create the save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if the dataset is already present
    if zip_path.exists() and not overwrite_existing:
        print(f"{zip_path} already exists. Skipping download.")
        return

    # Download the zip file
    print(f"Downloading {url} to {zip_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded {zip_path}")

    # Unpack the zip file
    print(f"Unpacking {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    print(f"Unpacked to {save_dir}")

    # Optionally, remove the zip file after unpacking
    os.remove(zip_path)
    print(f"Removed {zip_path}")
