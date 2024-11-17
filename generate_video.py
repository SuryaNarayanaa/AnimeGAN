import cv2
import numpy as np
import os

def crossfade_transition(img1, img2, alpha):
    """Apply crossfade transition between img1 and img2."""
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

def create_transition_video(image_folder, video_path, frame_rate=30, morph_frames=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    if len(images) < 2:
        raise ValueError("At least two images are required for morphing.")

    first_image_path = os.path.join(image_folder, images[0])
    last_image_path = os.path.join(image_folder, images[-1])

    img1 = cv2.imread(first_image_path)
    img2 = cv2.imread(last_image_path)

    height, width, _ = img1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    for i in range(len(images) - 1):
        img1_path = os.path.join(image_folder, images[i])
        img2_path = os.path.join(image_folder, images[i + 1])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        for j in range(morph_frames):
            alpha = j / float(morph_frames)
            crossfade_image = crossfade_transition(img1, img2, alpha)
            video_writer.write(crossfade_image)

    video_writer.release()
    print(f"Video saved as {video_path}")

if __name__ == "__main__":
    image_folder = "generated-4"  # Folder containing images
    video_path = "transition_video-2.mp4"  # Output video file
    frame_rate = 30  # Frames per second
    morph_frames = 30  # Number of frames per transition

    create_transition_video(image_folder, video_path, frame_rate, morph_frames)
