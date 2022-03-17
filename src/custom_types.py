import typing
import pathlib
import os

AllowedPathType = typing.Union[
    pathlib.Path,
    str,
    bytes,
    os.PathLike[bytes],
    None,
]
