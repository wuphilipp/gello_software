import os

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips
from moviepy.video.fx.speedx import speedx


def merge_videos(video_files, output_name, speed=1):
    videos = []
    for video_file in video_files:
        videos.append(VideoFileClip(video_file))
    final_clip = concatenate_videoclips(videos)
    # speed up
    final_clip = speedx(final_clip, speed)
    final_clip.write_videofile(output_name, fps=30)


def cv_render(img, name="GoalEnvExt", scale=5):
    """Take an image in ndarray format and show it with opencv."""
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:  # Depth. Normalize.
        img = np.tile(img, [1, 1, 3])
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)]
    h, w = new_img.shape[:2]
    new_img = cv2.resize(new_img, (w * scale, h * scale))
    cv2.imshow(name, new_img)
    cv2.waitKey(20)


def save_rgb(path, img):
    if np.max(img) <= 1.0:
        img = img * 255.0
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def make_grid(array, ncol=5, padding=0, pad_value=120):
    """numpy version of the make_grid function in torch. Dimension of array: NHWC"""
    if np.max(array) < 2.0:
        array = array * 255.0
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    if N % ncol > 0:
        res = ncol - N % ncol
        array = np.concatenate([array, np.ones([res, H, W, C])])
        N = array.shape[0]
    nrow = N // ncol
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(
            array[idx],
            [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]],
            constant_values=pad_value,
            mode="constant",
        )
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(
                array[idx],
                [[padding if i == 0 else 0, padding], [0, padding], [0, 0]],
                constant_values=pad_value,
                mode="constant",
            )
            row = np.hstack([row, cur_img])
        idx += 1
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img.astype(np.float32)


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    if np.max(array) <= 2.0:
        array *= 255.0
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + ".gif"

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def save_numpy_as_video(array, filename, fps=20):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    """
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if np.max(array) <= 2.0:
        array *= 255.0
    array = array.astype(np.uint8)
    # ensure that the file has the .mp4 extension
    fname, _ = os.path.splitext(filename)
    filename = fname + ".mp4"

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # import uuid
    # temp_filename = f'/tmp/{str(uuid.uuid4())}.mp4'
    # CV_VIDEO_CODES = {"mp4": cv2.VideoWriter_fourcc(*"mp4v"), }
    # img = array[0]
    # video_writer = cv2.VideoWriter(temp_filename, CV_VIDEO_CODES['mp4'], fps, (img.shape[1], img.shape[0]))
    #
    # # Save
    # for frame in list(array):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     video_writer.write(frame)
    # video_writer.release()
    # os.system(f"ffmpeg -i {temp_filename} -vcodec libx264 {filename} -y -hide_banner -loglevel error")
    # os.system(f"rm -rf {temp_filename}")\

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_videofile(filename, fps=fps, logger=None)
    return clip


def save_numpy_as_img(img, filename):
    img = img * 255.0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def save_numpy_to_gif_matplotlib(array, filename, interval=50):
    from matplotlib import animation
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    def img_show(i):
        plt.imshow(array[i])
        print("showing image {}".format(i))
        return

    ani = animation.FuncAnimation(fig, img_show, len(array), interval=interval)

    ani.save("{}.mp4".format(filename))

    import ffmpy

    ff = ffmpy.FFmpeg(
        inputs={"{}.mp4".format(filename): None},
        outputs={"{}.gif".format(filename): None},
    )

    ff.run()
    # plt.show()


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid_video_from_numpy(
    video_array, ncol, output_name="./output.mp4", speedup=1, fps=24
):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=5)

        # save_numpy_as_img(grid_frame / 255.0, output_name.replace('.mp4', f'_{t}.jpg'))

        grid_frames.append(grid_frame)

    save_numpy_as_video(np.array(grid_frames), output_name, fps=fps)


def make_grid_gif_from_numpy(
    video_array, ncol, output_name="./output.gif", speedup=1, fps=10
):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=5)
        grid_frames.append(grid_frame)
    save_numpy_as_gif(np.array(grid_frames), output_name, fps=fps)


def make_grid_video(video_list, ncol, output_name="./output.mp4", speedup=1):
    videos = []
    for video_path in video_list:
        myclip = VideoFileClip(video_path)
        if myclip.size[0] > 256:
            myclip = myclip.resize(height=256)
        if speedup != 1:
            myclip = myclip.speedx(speedup)
        frames = []
        for frame in myclip.iter_frames():
            frames.append(frame)
        videos.append(np.array(frames))
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=5)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, fps=24)


def visualize_traj_opencv(imgs):
    import cv2 as cv

    for i in range(len(imgs)):
        cv.imshow("x", imgs[i])
        cv.waitKey(20)
