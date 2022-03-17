from dataclasses import dataclass
import typing


@dataclass
class MatchParams:
    distance_threshold: typing.Union[float, int] = 0.75
    k_param: int = 2
    hough_bins: int = 2
    ransac_threshold: float = 5.0


@dataclass
class MatchedKeypoints:
    reference_keypoints: typing.Iterable = tuple()
    lookup_keypoints: typing.Iterable = tuple()
