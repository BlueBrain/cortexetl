import cv2
import blueetl
import pandas as pd
from glob import glob
from typing import List, Tuple


def images_from_filenames(list_of_filenames: List[str]) -> Tuple[List, Tuple[int, int]]:
    """
    Loads a list of image files and returns a list of image arrays and the size of each image.

    Args:
        list_of_filenames (List[str]): A list of image file paths.

    Returns:
        Tuple[List, Tuple[int, int]]: A tuple containing a list of image arrays and a tuple representing the size
        (width, height) of the images.

    Raises:
        TypeError: If the input list is not of type List[str].
        ValueError: If the input list is empty or if none of the images could be read.
    """

    # Check if the input is valid
    if not isinstance(list_of_filenames, list) or not all(
        isinstance(file, str) for file in list_of_filenames
    ):
        raise TypeError("Input list must be a list of file paths")

    # Check if the input list is empty
    if not list_of_filenames:
        raise ValueError("Input list is empty")

    img_array = []
    size = (0, 0)
    for filename in list_of_filenames:
        # Load image and check if it is not empty
        img = cv2.imread(filename)
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    # Check if at least one image was read successfully
    if not img_array:
        raise ValueError("None of the images could be read")

    return img_array, size


def video_from_image_files(list_of_filenames: List[str], output_filename: str) -> None:
    """
    Create a video file from a list of image files.

    Args:
    - list_of_filenames (List[str]): A list of filenames of the images to be used in the video.
    - output_filename (str): The filename (including path and extension) of the output video file.

    Returns:
    - None

    Raises:
    - TypeError: If list_of_filenames is not a list or output_filename is not a string.

    Note:
    - The function uses the OpenCV library to create the video file.
    - The output video file format is .mp4.
    - The video frames are written at a frame rate of 1 FPS.

    Example:
    ```
    list_of_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    video_from_image_files(list_of_files, 'output_video.mp4')
    ```
    """
    # Validate input types
    if not isinstance(list_of_filenames, list):
        raise TypeError("list_of_filenames should be a list of filenames")
    if not isinstance(output_filename, str):
        raise TypeError("output_filename should be a string")

    if list_of_filenames != []:
        img_array, size = images_from_filenames(list_of_filenames)

        if img_array != []:
            out = cv2.VideoWriter(
                output_filename, cv2.VideoWriter_fourcc(*"mp4v"), 1, size
            )
            for i in range(len(img_array)):
                frame = cv2.resize(img_array[i], size)
                out.write(frame)
            out.release()


def synapse_type_colors(synapse_type):
    """Return default the color of the synapse type."""
    if synapse_type == "EXC":
        return "darkred"
    elif synapse_type == "INH":
        return "blue"
    elif synapse_type == "ALL":
        return "black"


def creline_colors(creline):
    """Return default the color of the creline."""
    if creline == "PV":
        return "red"
    elif creline == "Htr3a":
        return "C0"
    elif creline == "SST":
        return "green"
