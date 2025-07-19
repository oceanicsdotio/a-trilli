from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image, ImageFont, ImageDraw
from numpy import uint8, random, array, fromfunction, roll
from time import time as now

WIDTH = 3840//2
HEIGHT = 2160//2
PREFIX = "random"
FEEDS = [3]
COLOR = (0, 0, 0, 127)
FPS = 24
MONOSPACE = ImageFont.truetype("Courier New Bold.ttf", HEIGHT)
CODEC = VideoWriter_fourcc(*"MJPG")
FORMAT = "avi"
DURATION = 5  # seconds, 
FRAME_SIZE = WIDTH * HEIGHT * 3
LOG_INTERVAL = 5  # frames
TEXT_UPDATE = 3  # frames
SHAPE = (HEIGHT, WIDTH, 3)
random.seed(42) # For reproducibility
NOISE = random.uniform(0, 8, size=SHAPE).astype(uint8) * 32

def partial_count_from_frames_and_feed(frame: int, feed: int) -> str:
    """
    Deterministically generate the 3 digits of text for breaking
    the pixel count across multiple videos.
    """
    count = (frame + 1 - (frame % TEXT_UPDATE)) * FRAME_SIZE
    text = str(count).zfill(12)
    offset = feed * 3
    return text[offset:offset+3]

def create_image(frame: int, feed: int):
    """
    Compose an image with of noise/pattern and text
    with part of the total count of pixels shown so far
    across 4 parallel videos.
    """
    def fragment(i, j, k) -> uint8:
        return roll(NOISE, frame * -1, axis=1)
    buffer = fromfunction(fragment, SHAPE, dtype=uint8)
    image = Image.fromarray(buffer)
    draw = ImageDraw.Draw(image)
    part = partial_count_from_frames_and_feed(frame, feed)
    draw.text((0, 0), part, COLOR, font=MONOSPACE)
    return array(image)

def create_video(
    feed: int,
    fps: int = FPS,
    duration: int = DURATION,
    start: int = 0,
    log_interval: int = LOG_INTERVAL
):
    """
    Render video for one of four parallel feeds.
    """
    clock = now()
    video = VideoWriter(f"frames/{PREFIX}_{feed}.{FORMAT}", CODEC, FPS, (WIDTH, HEIGHT))
    frames = fps * duration
    stop = start + frames
    for frame in range(start, stop):
        data = create_image(frame, feed)
        video.write(data)
        if log_interval is not None and frame % log_interval == log_interval - 1:
            print(f"Video {feed}, rendered frames {start+1}-{frame+1}/{stop}.")
    video.release()
    print(f"Finished in {now() - clock:.2f} seconds.")

if __name__ == "__main__":
    for feed in FEEDS:
        create_video(feed)


