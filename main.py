import os
import time

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

# import pydicom

from SelectCharacteristics import SelectCharacteristics
from ShowCharacteristics import ShowCharacteristics
from CanvasImage import CanvasImage
from Algorithms import Algorithms

class MyWindow:
    BIRADS_CLASSES = ["1", "2", "3", "4"]

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
        self.create_original_img_ui_elements()
        self.create_cropped_img_ui_elements()

        self.root.mainloop()

    def create_original_img_ui_elements(self):
        self.original_img_title = tk.Label(text="Original")
        self.original_img_title.grid(row=1, column=1, pady=15)

        self.root.rowconfigure(2, weight=1)  # make the CanvasImage widget expandable
        self.root.columnconfigure(1, weight=1)

        self.canvasimage = CanvasImage(self.root, self.original_img.filename)  # create widget
        self.canvasimage.bind("<Button-1>", self.original_img_click_event)
        self.canvasimage.grid(row=2, column=1)  # show widget

        self.original_img_label_text = tk.Label(text=self.original_img.filename)
        self.original_img_label_text.grid(row=3, column=1, padx=30)

    def create_cropped_img_ui_elements(self):
        tk_image = ImageTk.PhotoImage(self.cropped_img)

        self.cropped_img_title = tk.Label(text="Cropped")
        self.cropped_img_title.grid(row=1, column=2, pady=15)

        self.cropped_img_label = tk.Label(image=tk_image)
        self.cropped_img_label.photo_ref = tk_image
        self.cropped_img_label.grid(row=2, column=2, padx=30, pady=30)

        self.cropped_img_label_text = tk.Label(
            text=os.path.basename(self.cropped_img.filename))
        self.cropped_img_label_text.grid(row=3, column=2, padx=30)

    def load_original_image(
        self,
        image_path: str
    ):
        # if image_path[-4:] == '.dcm':
        #     self.original_img = pydicom.dcmread(image_path).pixel_array*128
        # else:

        self.original_img = Image.open(image_path)

        self.canvasimage = CanvasImage(self.root, image_path)  # create widget
        self.canvasimage.bind("<Button-1>", self.original_img_click_event)
        self.canvasimage.grid(row=2, column=1)  # show widget

        self.original_img_label_text.configure(text=os.path.basename(image_path))

    def load_cropped_image(self, image_path: str):
        self.cropped_img = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(self.cropped_img)

        self.cropped_img_label.configure(image=tk_image)
        self.cropped_img_label.photo_ref = tk_image

        self.cropped_img_label_text.configure(text=os.path.basename(image_path))

    def get_cropped_image_path(self):
        original_filename = os.path.basename(self.original_img.filename)
        name, extension = os.path.splitext(original_filename)
        return self.original_img.filename.replace(
            original_filename,
            f"{name}_cropped{extension}"
        )

    def crop_original_img(self, click_x, click_y):
        x = self.canvasimage.canvas.canvasx(click_x)
        y = self.canvasimage.canvas.canvasy(click_y)

        if self.canvasimage.outside(x, y): return

        rectangle_coords = self.canvasimage.get_drawing_rectangle_coords(click_x, click_y)

        return self.original_img.crop(rectangle_coords)

    def original_img_click_event(self, event: tk.Event):
        self.cropped_img = self.crop_original_img(event.x, event.y)
        if not self.cropped_img: return
        cropped_image_path = self.get_cropped_image_path()
        self.cropped_img.save(cropped_image_path)

        self.load_cropped_image(cropped_image_path)

    def open_image_file(self, initialdir: str = "imagens"):
        supported_formats = ["png", "tiff", "dcm"]
        image_path = filedialog.askopenfilename(
            initialdir=initialdir,
            title="Choose an imagem",
            filetypes=(
                ("all files", "*.*"),
                # ("png files", "*.png"),
                # ("tiff files", "*.tiff"),
                # ("dicom files", "*.dcm"),
            )
        )
        if not image_path or len(image_path) == 0: return
        name, extension = os.path.splitext(os.path.basename(image_path))
        if extension not in supported_formats: return
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

    def open_show_characteristics_window(self):
        start = time.time()
        descriptors = self.algorithms.get_all_image_descriptors(
            self.algorithms.resample_image(self.cropped_img)
        )
        ShowCharacteristics(
            self.root,
            time.time() - start,
            descriptors[0],
            descriptors[1],
            descriptors[2],
            descriptors[3],
            descriptors[4],
            descriptors[5],
            descriptors[6],
            descriptors[7],
            descriptors[8],
            descriptors[9],
            descriptors[10],
            descriptors[11],
            descriptors[12],
            [
                descriptors[13],
                descriptors[14],
                descriptors[15],
                descriptors[16],
                descriptors[17],
                descriptors[18],
                descriptors[19],
            ]
        )

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
            label="Calculate and show the characteristics for the selected region",
            command=self.open_show_characteristics_window
        )
        self.options_menu.add_command(
            label="classificar a imagem ou a região de interesse selecionada com o mouse",
            command=lambda: None
        )

        self.images_menu = tk.Menu(self.main_menu)
        self.main_menu.add_cascade(label="Images", menu=self.images_menu)

        for birads_class in self.BIRADS_CLASSES:
            self.images_menu.add_command(
                label=birads_class,
                command=lambda: self.open_image_file(
                    os.path.join("imagens", birads_class))
            )


MyWindow()
