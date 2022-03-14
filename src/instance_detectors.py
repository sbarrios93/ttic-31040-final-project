import cv2
import typing
import os
from pathlib import Path
import numpy as np
import typing
from dataclasses import dataclass


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

        self.reference_image: typing.Union[np.ndarray, None] = None
        self.reference_image_path: typing.Union[os.PathLike, None] = None
        self.reference_height: typing.Union[int, None] = None
        self.reference_width: typing.Union[int, None] = None
        self.reference_corners: typing.Union[np.ndarray, None] = None

        self.lookup_image: typing.Union[np.ndarray, None] = None
        self.lookup_image_path: typing.Union[os.PathLike, None] = None

        # set when running self.fit
        self.reference_keypoints = None
        self.reference_descriptors = None

        # set when running self.predict
        self.lookup_keypoints = None
        self.lookup_descriptors = None
        self.match_params = None

        # set when running _match_keypoints (by running parent function self.predict)
        self.unfiltered_matches = None

        # set when running _filter_keypoints (by running parent function self.predict)
        self.filtered_matches = None

        # check if fit has been called, theres some variables that are set to self when fit is ran
        # -> check the function for more detail, but the main ones are
        # -> self.sift
        # -> self.reference_keypoints
        # -> self.reference_descriptors
        self.fit_ran: bool = False

    def _register_reference_geometry(self):
        self.reference_height, self.reference_width = self.reference_image.shape[:2]
        self.reference_corners = np.array(
            [
                [0, 0],
                [0, self.reference_height],
                [self.reference_width, 0],
                [self.reference_width, self.reference_height],
            ]
        )

    def _check_and_sort_image(
        self,
        image: typing.Union[None, str, os.PathLike, np.ndarray],
        is_reference: bool = False,
        cv_color_profile=cv2.COLOR_BGR2RGB,
    ):
        """
        Checks if the image is of the right type
        and returns True if it is.

        self.reference_image and self.reference_image_path are set accordingly.
        """

        # check if image is None, if so throw error
        if image is None:
            raise ValueError(
                "Please input a reference image of any of types <str, os.PathLike, pathlib.Path, np.ndarray>"
            )

        # check validity of image type
        if isinstance(image, np.ndarray):
            if is_reference:
                self.reference_image_path = None
                self.reference_image = image
            else:
                self.lookup_image_path = None
                self.lookup_image = image
        elif isinstance(image, (str, os.PathLike)):  # note that pathlib.Path is also os.PathLike
            if is_reference:
                self.reference_image_path = Path(image)
                self.reference_image = cv2.cvtColor(cv2.imread(str(self.reference_image_path)), cv_color_profile)
            else:
                self.lookup_image_path = Path(image)
                self.lookup_image = cv2.cvtColor(cv2.imread(str(self.lookup_image_path)), cv_color_profile)
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
        self._check_and_sort_image(reference_image, is_reference=True, cv_color_profile=cv_color_profile)

        # init keypoints, descriptors
        self.reference_keypoints, self.reference_descriptors = self.sift.detectAndCompute(self.reference_image, None)

        # set fit ran to true after doing all steps in the function
        self.fit_ran = True

    @dataclass
    class MatchParams:
        distance_threshold: typing.Union[float, int] = 0.75
        k: int = 2
        min_match_count: int = 4

    def _register_params(self, **kwargs):
        # check if kwargs are valid
        for k, v in kwargs.items():
            if k not in dir(self.MatchParams):
                raise ValueError(
                    f"{k} is not a valid parameter for MatchParams, please use one of {list(self.MatchParams.__dataclass_fields__.keys())}. Parameters are passed through predict function as kwargs"
                )
            if v is None:
                print(
                    f"WARNING: {k} is set to None, this is not valid, please set a value for {k}. Setting to default value {self.MatchParams.__dataclass_fields__[k].default}"
                )
                continue

        self.match_params = self.MatchParams(**kwargs)

    def _match_keypoints(self):

        matches = cv2.BFMatcher().knnMatch(self.lookup_descriptors, self.reference_descriptors, k=self.match_params.k)
        return matches

    def _filter_keypoints(self):
        """
        Filters the keypoints by the distance threshold
        """
        # filter the keypoints by distance
        filtered_matches = []
        for m in self.unfiltered_matches:
            if m[0].distance < self.match_params.distance_threshold * m[1].distance:
                filtered_matches.append(m[0])

        return filtered_matches

    def _find_homography(self):

        assert len(self.filtered_matches) >= self.match_params.min_match_count, "Not enough matches"  # type: ignore
        # find the homography

        reference_points_homography = np.float32(
            [self.reference_keypoints[m.trainIdx].pt for m in self.filtered_matches]  # type: ignore
        )
        lookup_points_homography = np.float32(
            [self.lookup_keypoints[m.queryIdx].pt for m in self.filtered_matches]  # type: ignore
        )

        return reference_points_homography, lookup_points_homography

    def predict(self, lookup_image, cv_color_profile=cv2.COLOR_BGR2RGB, **kwargs):
        assert self.fit_ran, "Please call fit before calling predict"

        # register parameters as namedtuple on self.match_params
        self._register_params(**kwargs)

        # sort out the image depending on the type loaded
        self._check_and_sort_image(lookup_image, is_reference=False, cv_color_profile=cv_color_profile)

        # MATCHIN PIPELINE START MARKER
        # extract keypoints, descriptos from lookup_image
        self.lookup_keypoints, self.lookup_descriptors = self.sift.detectAndCompute(self.lookup_image, None)

        self.unfiltered_matches = self._match_keypoints()
        self.filtered_matches = self._filter_keypoints()

        self.H = self._find_homography()

        return self.H
