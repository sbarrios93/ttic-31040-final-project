from src.instance_detectors import Detector
from src.utils import Drawer
from pathlib import Path
import typing
import rich_click as click
from src.custom_dataclasses import MatchParams

# load rich_click arguments
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.USE_RICH_MARKUP = True

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
                click.UsageError(
                    f"{save_path.parent} does not exist, create it first and run again"
                )
            # if file exists, ask to overwrite
            if save_path.exists():
                click.confirm(f"{save_path} already exists, overwrite?", abort=True)
            # if path to lookup image is a dir, the save_path cannot be a file
            if path_to_lookup_image_or_dir.is_dir():
                click.UsageError(
                    f"{path_to_lookup_image_or_dir} is a directory, multiple files cant be save to a single file"
                )

        # if save_path is dir, make sure that the dir exists
        if save_path.is_dir():
            if not save_path.exists():
                click.UsageError(
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
)
@click.option(
    "save_format",
    "--save-format",
    "-f",
    type=click.Choice(["jpg", "png"]),
    default="jpg",
    help="File format to save result. Only works if --save-path does not link to a single file",
)
@click.option(
    "distance_threshold",
    "--distance-threshold",
    "-d",
    type=click.FloatRange(min=0, clamp=True),
    default=MatchParams.__dataclass_fields__["distance_threshold"].default,
    help="Distance Threshold",
)
@click.option(
    "k_param",
    "--k-param",
    "-k",
    type=click.IntRange(min=1, clamp=True),
    default=MatchParams.__dataclass_fields__["k_param"].default,
    help="k parameter",
)  # default params for the matching parameters are set on the MatchParams object
@click.option(
    "hough_bins",
    "--hough-bins",
    "-b",
    type=click.IntRange(min=1, clamp=True),
    default=MatchParams.__dataclass_fields__["hough_bins"].default,
    help="Hough Bin Count",
)
@click.option(
    "ransac_threshold",
    "--ransac-threshold",
    "-r",
    type=click.FloatRange(min=1.0, clamp=True),
    default=MatchParams.__dataclass_fields__["ransac_threshold"].default,
    help="RANSAC Threshold",
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
        click.UsageError("no images found")
    else:
        click.secho(f"Found {len(queue)} images to process", fg="green")

    # start detector fitting before looping, this only requires image in path_to_reference_image
    detector = Detector()
    detector.fit(path_to_reference_image)

    with click.progressbar(queue, label="Processing") as progress_bar:
        for path in progress_bar:
            _process_lookup_image(
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


if __name__ == "__main__":
    run_pipeline()
