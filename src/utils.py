import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.custom_types import AllowedPathType
from pathlib import Path
import rich_click as click


class Transformations:
    def __init__(self):
        pass

    def transform_keypoints_affine(
        self, lookup_keypoint: cv2.KeyPoint, reference_keypoint: cv2.KeyPoint
    ) -> typing.Tuple[
        np.ndarray[typing.Any, np.dtype[np.float64]],
        np.ndarray[typing.Any, np.dtype[np.float64]],
        typing.Union[float, int],
    ]:
        """
        Transforms the given keypoints from the lookup image to the reference image.
        Returns the transformed keypoints.
        """

        def _get_rotation_matrix(
            phi: typing.Union[float, int]
        ) -> np.ndarray[typing.Any, np.dtype[np.float64]]:
            """
            Returns the rotation matrix for the given angle.
            """
            phi = np.deg2rad(phi)

            rotation_matrix: np.ndarray[typing.Any, np.dtype[np.float64]] = np.array(
                [
                    [np.cos(phi), -np.sin(phi)],
                    [np.sin(phi), np.cos(phi)],
                ]
            )

            return rotation_matrix

        scale = lookup_keypoint.size / reference_keypoint.size
        rotation_matrix = _get_rotation_matrix(
            lookup_keypoint.angle - reference_keypoint.angle
        )

        point = (
            lookup_keypoint.pt
            + ((-1.0 * np.array(reference_keypoint.pt)) @ rotation_matrix.T) * scale
        )

        return (
            point,
            rotation_matrix,
            scale,
        )


class Drawer:
    def __init__(self, bbox_corners: np.ndarray, image: np.ndarray):
        self.bbox_corners = bbox_corners
        self.image = image

    def make_plot(
        self, show: bool, save: bool, save_dir: AllowedPathType, fig_name: str
    ):
        plt.close("all")
        plt.imshow(self.image)
        plt.axis("off")
        for i, point in enumerate(self.bbox_corners):
            # 4 corners, then iteration will be 0, 1, 2, 3, 0 with (i+1)%4
            xs = [self.bbox_corners[i][0], self.bbox_corners[(i + 1) % 4][0]]
            ys = [self.bbox_corners[i][1], self.bbox_corners[(i + 1) % 4][1]]
            plt.plot(xs, ys, "r")
        if save:
            plt.savefig(f"{str(save_dir)}/{fig_name}")
        if show:
            plt.show()


def _validate_args(path_to_lookup_image_or_dir, save_result, save_path):
    # manage argument errors
    if not save_result and save_path is not None:
        raise click.BadArgumentUsage("--save-path can only be used with --save")
    if save_path is None and save_result:
        raise click.BadParameter("--save-path must be specified")
    if save_result and (save_path is not None):
        save_path = Path(save_path).absolute()

        # if save_path is a file
        if save_path.is_file():
            #  make sure that the parent directory exists
            if not save_path.parent.exists():
                raise click.UsageError(
                    f"{save_path.parent} does not exist, create it first and run again"
                )
            # if file exists, ask to overwrite
            # if path to lookup image is a dir, the save_path cannot be a file
            if path_to_lookup_image_or_dir.is_dir():
                raise click.UsageError(
                    f"{path_to_lookup_image_or_dir} is a directory, multiple files cant be save to a single file"
                )
            if save_path.exists():
                click.confirm(f"{save_path} already exists, overwrite?", abort=True)

        # if save_path is dir, make sure that the dir exists
        if save_path.is_dir():
            if not save_path.exists():
                raise click.UsageError(
                    f"{save_path} does not exist, create it first and run again"
                )
    return save_path


def _build_lookup_queue(path_to_lookup_image_or_dir):
    queue: typing.List[Path] = []  # image processing queue
    if path_to_lookup_image_or_dir.is_file():
        queue.append(path_to_lookup_image_or_dir)
    if path_to_lookup_image_or_dir.is_dir():
        queue += list(path_to_lookup_image_or_dir.rglob("*"))

    # select only file that are images
    queue = [x for x in queue if x.is_file() and x.suffix in (".png", ".jpg", ".jpeg")]
    return queue


def _process_lookup_image(
    dont_display_result,
    save_result,
    save_path,
    save_format,
    k_param,
    distance_threshold,
    hough_bins,
    ransac_threshold,
    detector,
    path,
):
    bbox = detector.predict(
        path,
        k_param=k_param,
        distance_threshold=distance_threshold,
        hough_bins=hough_bins,
        ransac_threshold=ransac_threshold,
    )
    # display results
    if save_result:
        fig_dir = [save_path.parent if save_path.is_file() else save_path][0]
        fig_name_w_extension = [
            save_path.name if save_path.is_file() else path.stem + "." + save_format
        ][
            0
        ]  # if save_dir does not contain a filename in path, reuse filename from loaded path, append extension set in command line
    else:
        fig_dir = None
        fig_name_w_extension = None

    drawer = Drawer(bbox_corners=bbox, image=detector.lookup_image)

    collected_plotter_params = {
        "save": save_result,
        "show": (not dont_display_result),
        "save_dir": fig_dir,
        "fig_name": fig_name_w_extension,
    }

    drawer.make_plot(**collected_plotter_params)
