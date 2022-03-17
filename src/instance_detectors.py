import os
import typing
from pathlib import Path

import cv2
import numpy as np
from src.custom_dataclasses import MatchedKeypoints, MatchParams
from src.utils import Transformations
from src.custom_types import AllowedPathType
import rich_click as click


class Detector:
    """
    Shamelessly borrowing from the scikit-learn API:
    1. Fit: takes the reference image
    2. Predict: takes the frame where to find the instance and returns the bbox
    """

    def __init__(self):

        # initialize this right away
        # will be used when doing fit and predict
        self.sift = cv2.SIFT_create()

        self._reference_image: np.ndarray[typing.Any, np.dtype[np.float64]]
        self._reference_image_path: AllowedPathType
        self._reference_height: int
        self._reference_width: int
        self._reference_corners: np.ndarray[typing.Any, np.dtype[np.float64]]

        self.lookup_image: np.ndarray[typing.Any, np.dtype[np.float64]]
        self.lookup_image_path: AllowedPathType

        # set when running self.fit
        self._reference_keypoints: typing.Iterable[cv2.KeyPoint]
        self._reference_descriptors: np.ndarray[typing.Any, np.dtype[np.float64]]

        # set when running self.predict
        self._lookup_keypoints: typing.Iterable[cv2.KeyPoint]
        self._lookup_descriptors: np.ndarray[typing.Any, np.dtype[np.float64]]
        self._match_params: MatchParams

        # set when running _match_keypoints (by running parent function self.predict)
        self._unfiltered_matches: MatchedKeypoints

        # set when running _filter_keypoints (by running parent function self.predict)
        self._matched_keypoints: MatchedKeypoints

        # check if fit has been called, theres some variables that are set to self when fit is ran
        # -> check the function for more detail, but the main ones are
        # -> self.sift
        # -> self._reference_keypoints
        # -> self._reference_descriptors
        self._fit_ran: bool = False

        self._warped_keypoints_matched: np.ndarray[typing.Any, np.dtype[np.float64]]

    def _register_reference_geometry(self):
        self._reference_height, self._reference_width = self._reference_image.shape[:2]
        self._reference_corners = np.array(
            [
                [0, 0],
                [self._reference_width, 0],
                [self._reference_width, self._reference_height],
                [0, self._reference_height],
            ]
        )  # same as OpenCV

    def _check_and_sort_image(
        self,
        image: typing.Any,
        is_reference: bool = False,
        cv_color_profile: int = cv2.COLOR_BGR2RGB,
    ) -> bool:

        # check if image is None, if so throw error
        if image is None:
            raise ValueError(
                "Please input a reference image of any of types <str, os.PathLike, pathlib.Path, np.ndarray>"
            )

        # check validity of image type
        if isinstance(image, np.ndarray):
            if is_reference:
                self._reference_image_path = None
                self._reference_image = image
            else:
                self.lookup_image_path = None
                self.lookup_image = image
        elif isinstance(image, (str, os.PathLike)):
            if is_reference:
                self._reference_image_path = Path(image)
                self._reference_image = cv2.cvtColor(
                    cv2.imread(str(self._reference_image_path)), cv_color_profile
                )
            else:
                self.lookup_image_path = Path(image)
                self.lookup_image = cv2.cvtColor(
                    cv2.imread(str(self.lookup_image_path)), cv_color_profile
                )
        else:
            raise TypeError(
                f"Please input a {['reference image' if is_reference else 'image'][0]} image of any of types <str, os.PathLike, pathlib.Path, np.ndarray>, got {type(image)}"
            )

        # add the height and width of the reference image now
        if is_reference:
            self._register_reference_geometry()

        return True

    def fit(
        self,
        reference_image: typing.Union[None, str, os.PathLike, np.ndarray] = None,
        cv_color_profile=cv2.COLOR_BGR2RGB,
    ):
        # sort out the reference image depending on the type loaded
        self._check_and_sort_image(
            reference_image, is_reference=True, cv_color_profile=cv_color_profile
        )

        # init keypoints, descriptors
        (
            self._reference_keypoints,
            self._reference_descriptors,
        ) = self.sift.detectAndCompute(self._reference_image, None)

        # set fit ran to true after doing all steps in the function
        self._fit_ran = True

    def _register_params(self, **kwargs):
        # check if kwargs are valid
        for k, v in kwargs.items():
            if k not in dir(MatchParams):
                raise ValueError(
                    f"{k} is not a valid parameter for MatchParams, please use one of {list(MatchParams.__dataclass_fields__.keys())}. Parameters are passed through predict function as kwargs"
                )
            if v is None:
                click.secho(
                    f" WARNING: {k} is set to None, this is not valid, please set a value for {k}. Setting to default value {MatchParams.__dataclass_fields__[k].default}",
                    fg="bright_yellow",
                    # bold=True,
                    blink=True,
                )
                kwargs[k] = MatchParams.__dataclass_fields__[k].default
                continue

        self._match_params = MatchParams(**kwargs)

    def _match_keypoints(self) -> MatchedKeypoints:

        matches = cv2.BFMatcher().knnMatch(
            self._lookup_descriptors,
            self._reference_descriptors,
            k=self._match_params.k_param,
        )
        return matches

    def _filter_keypoints(self):
        """
        Filters the keypoints by the distance threshold
        """
        # filter the keypoints by distance
        filtered_matches: list[typing.Any] = []
        for m in self._unfiltered_matches:
            if m[0].distance < self._match_params.distance_threshold * m[1].distance:
                filtered_matches.append(m[0])

        matched_keypoints = MatchedKeypoints(
            reference_keypoints=[
                self._reference_keypoints[m.trainIdx] for m in filtered_matches
            ],
            lookup_keypoints=[
                self._lookup_keypoints[m.queryIdx] for m in filtered_matches
            ],
        )

        return matched_keypoints

    def _warp_keypoints(self):
        """
        Warps the lookup points to the reference points
        """

        stacked_keypoints: np.ndarray[typing.Any, np.dtype[np.float64]]
        warped_keypoints: np.ndarray[typing.Any, np.dtype[np.float64]]
        rotation_matrices: np.ndarray[typing.Any, np.dtype[np.float64]]

        reference_keypoints, lookup_keypoints = (
            self._matched_keypoints.reference_keypoints,
            self._matched_keypoints.lookup_keypoints,
        )

        stacked_keypoints = np.hstack(
            [
                np.array(reference_keypoints).reshape(-1, 1),
                np.array(lookup_keypoints).reshape(-1, 1),
            ]
        )  # shape (N,2)

        # transform all keypoints
        # transform_keypoints_affine returns (keypoint, rotation_matrix, scale) for each keypoint. we index the lambda return to [0] so we only get the keypoints
        warped_keypoints = np.apply_along_axis(
            lambda x: Transformations().transform_keypoints_affine(*x)[0],
            1,
            stacked_keypoints,
        )
        # [1] returns the rotation matrices
        rotation_matrices = np.apply_along_axis(
            lambda x: Transformations().transform_keypoints_affine(*x)[1],
            1,
            stacked_keypoints,
        )
        # [2] returns the scales
        scales = np.apply_along_axis(
            lambda x: Transformations().transform_keypoints_affine(*x)[2],
            1,
            stacked_keypoints,
        )

        return warped_keypoints, rotation_matrices, scales

    def _hough_voting(self):
        """
        Apply hough voting to the warped keypoints to get the final homography
        """

        votes = np.apply_along_axis(
            lambda x: np.digitize(x, np.histogram(x, self._match_params.hough_bins)[1]),
            1,
            self._warped_keypoints_matched,
        )

        # get voting stats
        _, votes_indices, votes_counts = np.unique(
            votes, axis=0, return_counts=True, return_inverse=True
        )

        # get indices of the vote that won
        winning_keypoints = np.where(votes_indices == votes_counts.argmax())

        reference_keypoints, lookup_keypoints = (
            self._matched_keypoints.reference_keypoints,
            self._matched_keypoints.lookup_keypoints,
        )

        return (
            np.array(reference_keypoints)[winning_keypoints[0].astype(int)],
            np.array(lookup_keypoints)[winning_keypoints[0].astype(int)],
        ), winning_keypoints

    def _find_homography(self):
        # find homography using ransac and cv2

        reference_points = np.asarray(
            [kp.pt for kp in self._hough_filtered_reference_keypoints]
        ).reshape(-1, 1, 2)
        lookup_points = np.asarray(
            [kp.pt for kp in self._hough_filtered_lookup_keypoints]
        ).reshape(-1, 1, 2)
        _homography, mask = cv2.findHomography(
            reference_points,
            lookup_points,
            cv2.RANSAC,
            self._match_params.ransac_threshold,
        )  # get the mask we need to filter out the bad matches, will do another homography with the good values of the mask

        # with indices where mask == 1, do homography again, this time with the good matches
        selected_idx = np.where(mask == 1)[0]
        H, _ = cv2.findHomography(
            reference_points[selected_idx],
            lookup_points[selected_idx],
            cv2.RANSAC,
            self._match_params.ransac_threshold,
        )

        return H

    def predict(
        self, lookup_image, cv_color_profile=cv2.COLOR_BGR2RGB, **kwargs
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        assert self._fit_ran, "Please call fit before calling predict"

        # register parameters as namedtuple on self.match_params
        self._register_params(**kwargs)

        # sort out the image depending on the type loaded
        self._check_and_sort_image(
            lookup_image, is_reference=False, cv_color_profile=cv_color_profile
        )

        # MATCHIN PIPELINE START MARKER
        # extract keypoints, descriptos from lookup_image
        self._lookup_keypoints, self._lookup_descriptors = self.sift.detectAndCompute(
            self.lookup_image, None
        )

        self._unfiltered_matches = self._match_keypoints()
        self._matched_keypoints = self._filter_keypoints()  # type: ignore

        (
            self._warped_keypoints_matched,
            _,
            _,
        ) = self._warp_keypoints()

        (
            (
                self._hough_filtered_reference_keypoints,
                self._hough_filtered_lookup_keypoints,
            ),
            self._hough_winning_keypoints_indices,
        ) = self._hough_voting()

        self.H = self._find_homography()  # transformation Homography

        # get bounding box for lookup image by perspective transform
        self._lookup_bounding_box: np.ndarray = cv2.perspectiveTransform(
            self._reference_corners.reshape(-1, 1, 2).astype(float), self.H
        ).reshape(4, 2)

        return self._reference_corners, self._lookup_bounding_box
