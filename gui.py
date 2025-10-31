import customtkinter as ctk
from utils.input_output import load_icon_images
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2


class AppWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Removal App")
        self.root.geometry("1500x700")
        self.root.configure(bg="#222222")
        ctk.set_appearance_mode("dark")  # Options: "light", "dark", "system"
        ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

        # Main grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=5)
        self.root.grid_columnconfigure(1, weight=1)

        # Left frame (video feed)
        self.left_frame = ctk.CTkFrame(self.root, fg_color="white", corner_radius=15)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.cap = cv2.VideoCapture(0)

        self.update_video()

        # Sidebar (filters)
        self.sidebar = ctk.CTkScrollableFrame(
            self.root,
            fg_color="#111",
            width=370,
            corner_radius=15
        )
        self.sidebar.grid(row=0, column=1, sticky="nswe", padx=(5, 5), pady=(5, 5))
        #self.sidebar.grid_propagate(False)

        self.pattern_buttons = []
        self.selected_button = None

        self.icon_images = load_icon_images("assets/backgrounds")

        self.add_sidebar_buttons()

        # Bottom bar (controls)
        self.bottom_bar = ctk.CTkFrame(self.root, fg_color="#1a1a1a", corner_radius=15)
        self.bottom_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self.add_bottom_buttons()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            img = img.resize((self.left_frame.winfo_width(), self.left_frame.winfo_height()))

            imgtk = ctk.CTkImage(light_image=img, dark_image=img,
                                 size=(self.left_frame.winfo_width(), self.left_frame.winfo_height()))

            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk

        self.video_label.after(30, self.update_video)

    # Sidebar buttons
    def add_sidebar_buttons(self):
        label = ctk.CTkLabel(self.sidebar, text="PATTERNS", font=ctk.CTkFont(size=20, weight="bold"))
        label.pack(pady=(20, 10))

        self.add_pattern_button("BLUR")
        for name, pil_img in self.icon_images:
            self.add_pattern_button(name, pil_img)

    def add_pattern_button(self, name, image=None):
        button_width = 280
        button_height = 150
        if image:
            max_width = button_width - 20
            max_height = button_height - 20
            img_w, img_h = image.size
            scale = min(max_width / img_w, max_height / img_h)
            new_size = (int(img_w * scale), int(img_h * scale))
            resized_image = image.resize(new_size)
            photo = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=new_size)

            button = ctk.CTkButton(
                self.sidebar,
                text=" ",  # tiny text so border renders
                image=photo,
                width=button_width,
                height=button_height,
                corner_radius=20,
                fg_color="#222",
                hover_color="#333",
                border_width=0,
                border_color="#222",  # initial color same as background
                command=lambda b=name: self.select_button(b),
            )
        else:
            button = ctk.CTkButton(
                self.sidebar,
                text=name,
                width=280,
                height=100,
                corner_radius=20,
                fg_color="#222",
                hover_color="#333",
                border_width=0,
                border_color="#222",
                command=lambda b=name: self.select_button(b)
            )

        button.pack(pady=10)
        button._name = name  # assign the identifier
        self.pattern_buttons.append(button)

    def select_button(self, name):
        for btn in self.pattern_buttons:
            btn.configure(border_width=0, border_color=btn.cget("fg_color"))

        for btn in self.pattern_buttons:
            if getattr(btn, "_name", None) == name:
                btn.configure(border_width=3, border_color="blue")
                self.selected_button = name
                break

    # Bottom bar buttons
    def add_bottom_buttons(self):
        button_style = {
            "corner_radius": 20,
            "height": 50,
            "width": 180,
            "font": ctk.CTkFont(size=14, weight="bold"),
        }

        # Left buttons
        ctk.CTkButton(
            self.bottom_bar, text="TAKE A PHOTO", **button_style
        ).pack(side="left", padx=20, pady=20)

        ctk.CTkButton(
            self.bottom_bar, text="START RECORDING", **button_style
        ).pack(side="left", padx=20, pady=20)

        # Right buttons

        ctk.CTkButton(
            self.bottom_bar, text="SAVE IMAGE/VIDEO", fg_color="#3a7", hover_color="#4c9", **button_style
        ).pack(side="right", padx=20, pady=20)

        ctk.CTkButton(
            self.bottom_bar, text="DISCARD", fg_color="#b33", hover_color="#d55", **button_style
        ).pack(side="right", padx=20, pady=20)

    def on_close(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

