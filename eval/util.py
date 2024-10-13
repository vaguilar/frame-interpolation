# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for frame interpolation on a set of video frames."""
import os
import shutil
from typing import Generator, Iterable, List, Optional

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'


def read_image(filename: str) -> np.ndarray:
  """Reads an sRgb 8-bit image.

  Args:
    filename: The input filename to read.

  Returns:
    A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_data = tf.io.read_file(filename)
  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


def write_image(filename: str, image: np.ndarray) -> None:
  """Writes a float32 3-channel RGB ndarray image, with colors in range [0..1].

  Args:
    filename: The output filename to save.
    image: A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_in_uint8_range = np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
  image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)

  extension = os.path.splitext(filename)[1]
  if extension == '.jpg':
    image_data = tf.io.encode_jpeg(image_in_uint8)
  else:
    image_data = tf.io.encode_png(image_in_uint8)
  tf.io.write_file(filename, image_data)


"""
frame1 and frame2 are 4d arrays of image batches
"""
def _recursive_generator(
    frames1: np.ndarray,
    frames2: np.ndarray,
    num_recursions: int,
    interpolator: interpolator_lib.Interpolator,
    frames_dir: str,
    indices1: List[int],
    indices2: List[int],
    bar: Optional[tqdm] = None,
) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input images 1 (4d image batches).
    frame2: Input images 2 (4d image batches).
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    # remove batch dimension and write images
    a = tf.unstack(frames1, axis=0)
    b = tf.unstack(frames2, axis=0)
    for idx, frame in zip(indices1, a):
      write_image(f'{frames_dir}/frame_{idx:05d}.png', frame)
    for idx, frame in zip(indices2, b):
      write_image(f'{frames_dir}/frame_{idx:05d}.png', frame)
  else:
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frames = interpolator(frames1, frames2, time)
    bar.update(frames1.shape[0]) if bar is not None else bar
    yield from _recursive_generator(
        frames1,
        mid_frames,
        num_recursions - 1,
        interpolator,
        frames_dir,
        indices1,
        [(i+j)//2 for i, j in zip(indices1, indices2)],
        bar
    )
    yield from _recursive_generator(
        mid_frames,
        frames2,
        num_recursions - 1,
        interpolator,
        frames_dir,
        [(i+j)//2 for i, j in zip(indices1, indices2)],
        indices2,
        bar
    )


def interpolate_recursively_from_files(
    frames: List[str], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator,
    frames_dir: str,
    batch_size: int = 1,
) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Loads the files on demand and uses the yield paradigm to return the frames
  to allow streamed processing of longer videos.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for chunk_start in range(1, n, batch_size):
    end = min(n-1, chunk_start+batch_size)
    chunk = [chunk_start - 1] + list(range(chunk_start, end))
    if len(chunk) < 2:
      break
    chunked_frames = [read_image(frames[i]) for i in chunk]
    first_frames = tf.stack(chunked_frames[:-1], axis=0)
    second_frames = tf.stack(chunked_frames[1:], axis=0)
    offset = 2 ** batch_size
    yield from _recursive_generator(
        first_frames,
        second_frames,
        times_to_interpolate,
        interpolator,
        frames_dir,
        [i * offset for i in chunk[:-1]],
        [i * offset for i in chunk[1:]],
        bar
    )
  # Separately yield the final frame.
  yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  This is functionally equivalent to interpolate_recursively_from_files(), but
  expects the inputs frames in memory, instead of loading them on demand.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(1, n):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    times_to_interpolate, interpolator, bar)
  # Separately yield the final frame.
  yield frames[-1]


def get_ffmpeg_path() -> str:
  path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
  if not path:
    raise RuntimeError(
        f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' is not found;"
        " perhaps install ffmpeg using 'apt-get install ffmpeg'.")
  return path
