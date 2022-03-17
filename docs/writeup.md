---
title: Object Detection Writeup
subtitle: TTIC 31040
author: Sebastian Barrios
date: 2022/03/17
---

# Usage

The system works as a command line interface tool (CLI) to detect the object of interest in an image.

After installing the required dependencies located in `pyproject.toml`, run the following command to execute the program:

```bash
python -m main [OPTIONS] PATH_TO_REFERENCE_IMAGE PATH_TO_LOOKUP_IMAGE_OR_DIR
```

where the `PATH_TO_REFERENCE_IMAGE` is a path to a single image file and the `PATH_TO_LOOKUP_IMAGE_OR_DIR` is a path to a directory containing images or a path to a single image file. The images in the `PATH_TO_LOOKUP_IMAGE_OR_DIR` are the ones where the detector will try to draw a bounding  to detect the object of interest passed on the `PATH_TO_REFERENCE_IMAGE`.


When running with only the required arguments, the object will display the bounding box for each image on a separate window, without saving the results and prints the tuple of bounding boxes.


The detector also takes the following arguments as options:

1. `--no-display` Do not display result on external window
2. `--print` To print to console each reference and lookup bounding box
3. `--save` Save result to image file
4. `--save-path` Path to save result, only works if `--save` is specified. It can point to a file or it can be a directory, in which case the name of the image will be used as the filename
5. `--save-format` File format to save result. Only works if --save-path does not link to a single file. Takes `jpg` and `png` as options.
6. `--distance-threshold` Distance Threshold `[float with default: 0.75]`
7. `--k-param` k parameter `[integer with default: 2]`
8. `--hough-bins` Hough Bin Count `[integer with default: 2]`
9. `--ransac-threshold` RANSAC Threshold `[float with default: 5.0]`

# Description of the implementation



