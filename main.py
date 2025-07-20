from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image, ImageFont, ImageDraw
from numpy import uint8, random, array, roll
from time import time as now
from math import ceil

def partial_count_from_frames_and_feed(
    frame: int,
    feed: int,
    pixels_per_frame: int,
    debounce: bool = True,
    text_update: int = 15
) -> str:
    """
    Deterministically generate the 3 digits of text for breaking
    the pixel count across multiple videos.
    """
    pixels = (frame + 1 - (frame % text_update) * debounce) * pixels_per_frame
    text = str(pixels).zfill(12)
    offset = feed * 3
    return text[offset:offset+3]

def create_image(
    frame: int,
    feed: int,
    noise: array,
    font: ImageFont,
    pixels_per_frame: int,
    frame_size: tuple,
    font_color: tuple = (0, 0, 0),
    roll_factor: int = 4,
    resample: Image.Resampling = Image.Resampling.NEAREST,
    debounce: bool = True
) -> array:
    """
    Compose an image with of noise/pattern and text
    with part of the total count of pixels shown so far
    across 4 parallel videos.
    """
    offset = (-1 * frame) // roll_factor
    rolled = roll(noise, offset, axis=1)
    image = Image.fromarray(rolled).resize(frame_size, resample=resample)
    draw = ImageDraw.Draw(image)
    part = partial_count_from_frames_and_feed(frame, feed, pixels_per_frame=pixels_per_frame, debounce=debounce)
    draw.text((0, 0), part, font_color, font=font)
    return array(image)

def create_color_noise(
    frame_size: tuple,
    seed: int = 42,
    size_factor: int = 40
) -> array:
    """
    Create a noise image to be used as a base for the video frames.
    """
    random.seed(seed) # For reproducibility
    width, height = frame_size
    size = (height // size_factor, width // size_factor, 3)
    return random.uniform(0, 255, size=size).astype(uint8)

def create_video(
    feed: int,
    fps: int,
    frames: int,
    frame_size: tuple,
    pixels_per_frame: int,
    log_interval: int = 10,
    format: str = "mp4",
    four_cc: str = "avc1",
    prefix: str = "video/a-trilli",
    font: str = "Courier New Bold.ttf"
):
    """
    Render video for one of four parallel feeds.
    """
    clock = now()
    filename = f"{prefix}-{feed}.{format}"
    four_cc_int = VideoWriter_fourcc(*four_cc)
    _, height = frame_size
    true_type_font = ImageFont.truetype(font, height)
    video = VideoWriter(filename, four_cc_int, fps, frame_size)
    noise = create_color_noise(frame_size=frame_size)
    for frame in range(frames):
        # Final frame should overflow
        debounce = frame + 1 == frames
        img = create_image(
            frame, 
            feed, 
            noise, 
            pixels_per_frame=pixels_per_frame,
            frame_size=frame_size,
            debounce=debounce, 
            font=true_type_font
        )
        video.write(img)
        if log_interval is not None and frame % log_interval == log_interval - 1:
            print(f"Video {feed}, rendered frames {frame+1}/{frames}.")
    video.release()
    print(f"Finished in {now() - clock:.2f} seconds.")

if __name__ == "__main__":
    total_pixels = 1E12
    width = 3840
    height = 2160
    feeds = 4
    pixels_per_frame = width * height * feeds * 3
    create_video(
        feed=0, 
        fps=60, 
        frames=ceil(total_pixels / pixels_per_frame), 
        pixels_per_frame=pixels_per_frame, 
        frame_size=(width, height)
    )
