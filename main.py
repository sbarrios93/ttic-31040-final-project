from src.instance_detectors import Detector
from src.utils import _validate_args, _build_lookup_queue, _process_lookup_image
from pathlib import Path
import typing
import rich_click as click
from src.custom_dataclasses import MatchParams
from rich import print

# load rich_click arguments
click.rich_click.OPTION_GROUPS = {
    "main.py": [
        {
            "name": "Required Arguments",
            "options": [
                "path_to_reference_image",
                "path_to_lookup_image_or_dir",
            ],
        },
        {
            "name": "Options: Algorithm ",
            "options": [
                "--k-param",
                "--hough-bins",
                "--ransac-threshold",
                "--distance-threshold",
            ],
        },
        {
            "name": "Options: Visualization",
            "options": ["--no-display", "--save", "--save-path", "--save-format"],
        },
    ]
}
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.USE_RICH_MARKUP = True


@click.command()
@click.argument(
    "path_to_reference_image",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
)
@click.argument(
    "path_to_lookup_image_or_dir",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
)
@click.option(
    "dont_display_result",
    "--no-display",
    "-n",
    is_flag=True,
    default=False,
    help="Do not display result on external window",
)
@click.option(
    "print_bboxes",
    "--print",
    "-e",  # for echo
    is_flag=True,
    default=False,
    help="print the bboxes",
)
@click.option(
    "save_result",
    "--save",
    "-s",
    is_flag=True,
    default=False,
    help="Save result to file",
)
@click.option(
    "save_path",
    "--save-path",
    "-p",
    type=click.Path(exists=False, dir_okay=True, file_okay=True),
    default=None,
    help="Path to save result, only works if --save is specified. It can point to a file or it can be a directory, in which case the name of the image will be used as the filename.",
    show_default=True,
)
@click.option(
    "save_format",
    "--save-format",
    "-f",
    type=click.Choice(["jpg", "png"]),
    default="jpg",
    help="File format to save result. Only works if --save-path does not link to a single file",
    show_default=True,
)
@click.option(
    "distance_threshold",
    "--distance-threshold",
    "-d",
    type=click.FloatRange(min=0, clamp=True),
    default=MatchParams.__dataclass_fields__["distance_threshold"].default,
    help="Distance Threshold",
    show_default=True,
)
@click.option(
    "k_param",
    "--k-param",
    "-k",
    type=click.IntRange(min=1, clamp=True),
    default=MatchParams.__dataclass_fields__["k_param"].default,
    help="k parameter",
    show_default=True,
)  # default params for the matching parameters are set on the MatchParams object
@click.option(
    "hough_bins",
    "--hough-bins",
    "-b",
    type=click.IntRange(min=1, clamp=True),
    default=MatchParams.__dataclass_fields__["hough_bins"].default,
    help="Hough Bin Count",
    show_default=True,
)
@click.option(
    "ransac_threshold",
    "--ransac-threshold",
    "-r",
    type=click.FloatRange(min=1.0, clamp=True),
    default=MatchParams.__dataclass_fields__["ransac_threshold"].default,
    help="RANSAC Threshold",
    show_default=True,
)
def run_pipeline(
    path_to_reference_image: typing.Union[str, Path],
    path_to_lookup_image_or_dir: typing.Union[str, Path],
    dont_display_result: bool,
    save_result: bool,
    save_path: typing.Any,
    save_format: str,
    k_param: int,
    distance_threshold: typing.Union[float, int],
    hough_bins: int,
    ransac_threshold: float,
    print_bboxes: bool,
):
    """
    Arguments:\n
        [white][b]path_to_reference_image:[/][/] Path to reference image. May only be a path to a single image \n

        [b]path_to_lookup_image_or_dir:[/] Path to lookup image or directory. Can be a path to a single image or a directory of images. If a directory is given, then all images in the directory will be parsed
    """

    path_to_lookup_image_or_dir = Path(path_to_lookup_image_or_dir).absolute()
    path_to_reference_image = Path(path_to_reference_image).absolute()

    save_path = _validate_args(path_to_lookup_image_or_dir, save_result, save_path)

    queue = _build_lookup_queue(path_to_lookup_image_or_dir)

    if len(queue) == 0:
        raise click.UsageError("no images found")
    else:
        click.secho(f"Found {len(queue)} images to process", fg="green")

    # start detector fitting before looping, this only requires image in path_to_reference_image
    detector = Detector()
    detector.fit(path_to_reference_image)

    all_bboxes: list = []

    with click.progressbar(queue, label="Processing") as progress_bar:
        for path in progress_bar:
            reference_bbox, lookup_bbox = _process_lookup_image(
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
            )

            if print_bboxes:
                print(
                    f"\n[u]{path}:[/]\n [b]-> Reference:[/]\n {reference_bbox}\n [b]-> Lookup:[/]\n {lookup_bbox}"
                )
            all_bboxes.append((reference_bbox, lookup_bbox))
    return all_bboxes


if __name__ == "__main__":
    run_pipeline()
