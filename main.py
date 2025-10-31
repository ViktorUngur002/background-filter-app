import tkinter as tk
from gui import AppWindow

if __name__ == "__main__":
    root = tk.Tk()
    app = AppWindow(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()