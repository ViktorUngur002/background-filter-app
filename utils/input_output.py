import os
from PIL import Image
from tkinter import filedialog

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
