import os
from PIL import Image
from tkinter import filedialog
import cv2

def load_icon_images(directory_path, size=(260,100)):
    images = []
    if not os.path.exists(directory_path):
        print("[ERROR] Directory does not exist")
        return images

    for file in os.listdir(directory_path):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(directory_path, file)
            try:
                img = Image.open(img_path).resize(size)
                images.append((os.path.splitext(file)[0], img))
            except Exception as e:
                print("[ERROR] File not exist")

    return images

def save_image(image):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    if file_path:
        img = Image.fromarray(image)
        img.save(file_path)

def save_video(video, fps = 30):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )

    if file_path:
        height, width, _ = video[0].shape

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in video:
            # Convert RGB -> BGR before writing
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
