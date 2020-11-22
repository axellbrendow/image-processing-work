import os
from typing import List, Dict

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

from SelectCharacteristics import SelectCharacteristics
from CanvasImage import CanvasImage
from Algorithms import Algorithms

class MyWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Image Classifier")
        self.root.geometry("1024x650")

        self.algorithms = Algorithms()
        self.algorithms.load_images(show_msg_box=False)
        # self.algorithms.train()
        self.original_img = self.algorithms.images["1"][0]
        self.cropped_img = self.algorithms.images["1"][0]

        self.config_menu()
        self.load_images()

        self.original_img = self.images["1"][0]
        self.cropped_img = self.images["2"][0]

        self.original_img_title = tk.Label(text="Original")
        self.original_img_title.grid(row=1, column=1, pady=15)
        self.root.rowconfigure(2, weight=1)  # make the CanvasImage widget expandable
        self.root.columnconfigure(1, weight=1)
        self.canvasimage = CanvasImage(self.root, self.original_img.filename)  # create widget
        self.canvasimage.bind("<Button-1>", self.original_img_click_event)
        self.canvasimage.grid(row=2, column=1)  # show widget
        self.original_img_label_text = tk.Label(text=self.original_img.filename)
        self.original_img_label_text.grid(row=3, column=1, padx=30)

        self.cropped_img_title = tk.Label(text="Cortada")
        self.cropped_img_title.grid(row=1, column=2, pady=15)
        self.cropped_img_label = tk.Label(image=ImageTk.PhotoImage(self.cropped_img))
        self.cropped_img_label.grid(row=2, column=2, padx=30, pady=30)
        self.cropped_img_label.bind("<Button-1>", print)
        self.cropped_img_label_text = tk.Label(text=self.cropped_img.filename)
        self.cropped_img_label_text.grid(row=3, column=2, padx=30)

        self.root.mainloop()

    def load_original_image(
        self,
        image_path: str
    ):
        self.original_img = Image.open(image_path)

        self.canvasimage = CanvasImage(self.root, image_path)  # create widget
        self.canvasimage.bind("<Button-1>", self.original_img_click_event)
        self.canvasimage.grid(row=2, column=1)  # show widget

        self.original_img_label_text.configure(text=os.path.basename(image_path))

    def load_cropped_image(
        self,
        image_path: str,
        img_label: tk.Label,
        img_text_label: tk.Label
    ):
        image = Image.open(image_path)
        self.cropped_img = image
        tk_image = ImageTk.PhotoImage(image)
        img_label.configure(image=tk_image)
        img_label.photo_ref = tk_image
        img_text_label.configure(text=os.path.basename(image_path))

    def original_img_click_event(self, event: tk.Event):
        original_filename = os.path.basename(self.original_img.filename)
        name, extension = os.path.splitext(original_filename)
        cropped_image_path = self.original_img.filename.replace(
            original_filename,
            f"{name}_cropped{extension}"
        )
        x = self.canvasimage.canvas.canvasx(event.x)
        y = self.canvasimage.canvas.canvasy(event.y)

        if self.canvasimage.outside(x, y): return

        box_image = self.canvasimage.canvas.coords(self.canvasimage.container)

        x_offset = x - box_image[0] # offset from image corner inside canvas
        y_offset = y - box_image[1]
        
        x0 = max(0, x_offset - 64)
        y0 = max(0, y_offset - 64)
        x1 = min(self.original_img.width, x_offset + 64)
        y1 = min(self.original_img.height, y_offset + 64)

        self.cropped_img = self.original_img.crop((x0, y0, x1, y1))
        self.cropped_img.save(cropped_image_path)

        self.load_cropped_image(
            cropped_image_path,
            self.cropped_img_label,
            self.cropped_img_label_text
        )

    def open_image_file(self, initialdir: str = "imagens"):
        image_path = filedialog.askopenfilename(
            initialdir=initialdir,
            title="Escolha uma imagem",
            filetypes=(
                ("png files", "*.png"),
                ("tiff files", "*.tiff"),
                ("dicom files", "*.dcm"),
            )
        )
        if (len(image_path) == 0): return
        self.load_original_image(image_path)

    def set_used_descriptors(self):
        window = self.select_characteristics_window
        self.algorithms.set_used_descriptors(
            window.energyCheckVar.get() == 1,
            window.contrastVar.get() == 1,
            window.correlationVar.get() == 1,
            window.varianceVar.get() == 1,
            window.homogeneityVar.get() == 1,
            window.sumAverageVar.get() == 1,
            window.sumVarianceVar.get() == 1,
            window.sumEntropyVar.get() == 1,
            window.entropyVar.get() == 1,
            window.differenceVarianceVar.get() == 1,
            window.differenceEntropyVar.get() == 1,
            window.informationMeasuresOfCorrelation12Var.get() == 1,
            window.informationMeasuresOfCorrelation13Var.get() == 1,
            window.sevenInvariantHuMomentsVar.get() == 1,
        )
        window.root.destroy()
        window.root.update()

    def open_select_characteristics_window(self):
        new_window = SelectCharacteristics(self.root)
        self.select_characteristics_window = new_window
        self.select_characteristics_window.root.protocol(
            "WM_DELETE_WINDOW",
            self.set_used_descriptors
        )

    def load_images(self):
        if not os.path.exists('imagens'):
            messagebox.showinfo('Erro', 'A pasta imagens não foi encontrada')
            exit(0)

        self.images: Dict[str, List[Image.Image]] = {
            "1": [],
            "2": [],
            "3": [],
            "4": []
        }
        self.load_images_from_dir("1")
        self.load_images_from_dir("2")
        self.load_images_from_dir("3")
        self.load_images_from_dir("4")

    def config_menu(self):
        self.main_menu = tk.Menu(self.root)
        self.root.config(menu=self.main_menu)

        self.file_menu = tk.Menu(self.main_menu)
        self.main_menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_image_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        self.options_menu = tk.Menu(self.main_menu)
        self.main_menu.add_cascade(label="Options", menu=self.options_menu)
        self.options_menu.add_command(
            label="Read train/test images directory",
            command=self.algorithms.load_images
        )
        self.options_menu.add_command(
            label="Select the characteristics to use",
            command=self.open_select_characteristics_window
        )
        self.options_menu.add_command(
            label="treinar o classificador",
            command=lambda: None
        )
        self.options_menu.add_command(
            label="calcular e exibir as características para a imagem visualizada ou área selecionada",
            command=lambda: None
        )
        self.options_menu.add_command(
            label="classificar a imagem ou a região de interesse selecionada com o mouse",
            command=lambda: None
        )

        self.images_menu = tk.Menu(self.main_menu)
        self.main_menu.add_cascade(label="Imagens", menu=self.images_menu)


MyWindow()
