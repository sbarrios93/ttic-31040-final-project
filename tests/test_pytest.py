import pytest
from pathlib import Path
import numpy as np
from src.instance_detectors import Detector
import cv2


IMGS_FOR_TEST = [
    "./tests/images/stop1.jpg",
    Path("./tests/images/stop1.jpg"),
    cv2.cvtColor(cv2.imread("./tests/images/stop1.jpg"), cv2.COLOR_BGR2RGB),
]

IMGS_IDS_FOR_TEST = [str(type(x)) for x in IMGS_FOR_TEST]


@pytest.fixture
def valid_image(request):
    # init instance
    detector = Detector()
    return (request.param, detector)


@pytest.mark.parametrize("valid_image", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_reference_image_type(valid_image):
    image, detector = valid_image
    detector.fit(reference_image=image)


@pytest.mark.parametrize("valid_image", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_check_and_sort_image_as_reference(valid_image, reference=True):
    image, detector = valid_image
    assert detector._check_and_sort_image(image=image, is_reference=reference)


@pytest.mark.parametrize("valid_image", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_correct_variable_setting_reference_image(valid_image):
    image, detector = valid_image
    detector._check_and_sort_image(image=image, is_reference=True)
    assert isinstance(detector.reference_image, np.ndarray)

    if isinstance(image, np.ndarray):
        assert detector.reference_image_path is None
    else:
        assert detector.reference_image_path is not None


@pytest.mark.parametrize("valid_image", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_check_and_sort_image_as_lookup(valid_image):
    image, detector = valid_image

    detector._check_and_sort_image(image=image, is_reference=False)
    assert isinstance(detector.lookup_image, np.ndarray)

    if isinstance(image, np.ndarray):
        assert detector.lookup_image_path is None
    else:
        assert detector.lookup_image_path is not None


@pytest.fixture
def fitted_detector(request):
    image = request.param
    detector = Detector()
    detector.fit(reference_image=image)
    return detector


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_keypoints_exist_in_reference(fitted_detector):
    assert fitted_detector.reference_keypoints is not None
    assert len(fitted_detector.reference_keypoints) > 0


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_keypoints_are_Keypoints_class_in_reference(fitted_detector):
    assert all(isinstance(kp, cv2.KeyPoint) for kp in fitted_detector.reference_keypoints)


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_descriptors_exist_in_reference(fitted_detector):
    assert isinstance(fitted_detector.reference_descriptors, np.ndarray)


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_keypoints_exist_in_lookup(fitted_detector):
    fitted_detector.predict(lookup_image=fitted_detector.reference_image)
    assert fitted_detector.lookup_keypoints is not None
    assert len(fitted_detector.lookup_keypoints) > 0


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_keypoints_are_Keypoints_class_in_lookup(fitted_detector):
    fitted_detector.predict(lookup_image=fitted_detector.reference_image)
    assert all(isinstance(kp, cv2.KeyPoint) for kp in fitted_detector.lookup_keypoints)  # type: ignore


@pytest.mark.parametrize("fitted_detector", IMGS_FOR_TEST, ids=IMGS_IDS_FOR_TEST, indirect=True)
def test_descriptors_exist_in_lookup(fitted_detector):
    fitted_detector.predict(lookup_image=fitted_detector.reference_image)
    assert isinstance(fitted_detector.lookup_descriptors, np.ndarray)
