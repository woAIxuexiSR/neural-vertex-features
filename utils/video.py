import cv2
import numpy as np
import time


class Video:
    def __init__(self, width, height, fps, filename):

        self.writer = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)

        self.current = time.time()
        self.duration = 0
        self.frame = 0

    def __del__(self):
        self.writer.release()

    # img: (height, width, 3) np.uint8
    def write_img(self, img):

        self.writer.write(img)

        t = time.time()
        self.duration += t - self.current
        self.current = t
        self.frame += 1

    def real_fps(self):
        return self.frame / self.duration


if __name__ == "__main__":

    width, height = 1280, 720
    frame = 60

    img = np.zeros((height, width, 3), dtype=np.float32)
    x = np.arange(0, 1, 1 / width)
    y = np.arange(0, 1, 1 / height)
    xx, yy = np.meshgrid(x, y)

    out = Video(width, height, 30, "test.mp4")

    for i in range(frame):

        idx = i / frame

        img[:, :, 0] = xx * idx + (1 - xx) * (1 - idx)
        img[:, :, 1] = yy * idx + (1 - yy) * (1 - idx)
        img[:, :, 2] = 0.5

        out.write_img((img * 255).astype(np.uint8))

    print(out.real_fps())
