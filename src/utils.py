import numpy as np
import typing
import cv2
import matplotlib.pyplot as plt
from src.custom_types import AllowedPathType
from pathlib import Path


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

        def _get_rotation_matrix(phi: typing.Union[float, int]) -> np.ndarray[typing.Any, np.dtype[np.float64]]:
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
        rotation_matrix = _get_rotation_matrix(lookup_keypoint.angle - reference_keypoint.angle)

        point = lookup_keypoint.pt + ((-1.0 * np.array(reference_keypoint.pt)) @ rotation_matrix.T) * scale

        return (
            point,
            rotation_matrix,
            scale,
        )


class Drawer:
    def __init__(self, bbox_corners: np.ndarray, image: np.ndarray):
        self.bbox_corners = bbox_corners
        self.image = image

    def make_plot(self, show: bool, save: bool, save_dir: AllowedPathType, fig_name: str):
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

    # # def plot_on_screen(self):
    # #     return self.make_plot()

    # def plot_save(self, save_dir: AllowedPathType, fig_name: str):
    #     self.make_plot().savefig(f"{str(save_dir)}/{fig_name}")
