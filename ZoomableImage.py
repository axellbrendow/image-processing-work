import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class ZoomableImage:
    ''' Simple zoom with mouse wheel '''
    def __init__(self, mainframe, path):
        ''' Initialize the main Frame '''
        self.__imframe = ttk.Frame(mainframe)
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Open image
        self.image = Image.open(path)
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        vbar.configure(command=self.canvas.yview)  # bind scrollbars to the canvas
        hbar.configure(command=self.canvas.xview)
        # Make the canvas expandable
        self.__imframe.rowconfigure(0, weight=1)
        self.__imframe.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<Motion>', self.__draw_rectangle)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.__drawed_rectangle = None
        self.imscale = 1.0
        self.imageid = None
        self.container = None
        self.delta = 0.75
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def destroy(self):
        """ ImageFrame destructor """
        self.image.close()
        self.canvas.destroy()
        self.canvas.update()
        self.__imframe.destroy()
        self.__imframe.update()

    def bind(self, *args):
        self.canvas.bind(*args)

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def get_drawing_rectangle_x_coords(self, click_x_on_canvas, image_box):
        # offset from x of the upper left image corner inside canvas
        x_offset = click_x_on_canvas - image_box[0]

        # x of the upper left rectangle corner on the image
        rec_x0 = math.floor(x_offset - (64 * self.imscale))
        # x of the bottom right rectangle corner on the image
        rec_x1 = math.floor(x_offset + (64 * self.imscale))

        if rec_x0 < 0:  # checks if the rectangle is coming out on the left
            offset = -rec_x0
            rec_x0 += offset
            rec_x1 += offset
        elif rec_x1 > self.resized_image.width:  # checks if the rectangle is coming out on the right
            offset = rec_x1 - self.resized_image.width
            rec_x0 -= offset
            rec_x1 -= offset

        return (rec_x0, rec_x1)

    def get_drawing_rectangle_y_coords(self, click_y_on_canvas, image_box):
        # offset from y of the upper left image corner inside canvas
        y_offset = click_y_on_canvas - image_box[1]

        # y of the upper left rectangle corner on the image
        rec_y0 = math.floor(y_offset - (64 * self.imscale))
        # y of the bottom right rectangle corner on the image
        rec_y1 = math.floor(y_offset + (64 * self.imscale))

        if rec_y0 < 0:  # checks if the rectangle is coming out over the top
            offset = -rec_y0
            rec_y0 += offset
            rec_y1 += offset
        # checks if the rectangle is coming out over the bottom
        elif rec_y1 > self.resized_image.height:
            offset = rec_y1 - self.resized_image.height
            rec_y0 -= offset
            rec_y1 -= offset

        return (rec_y0, rec_y1)

    def get_drawing_rectangle_coords(self, mouse_x, mouse_y):
        x = self.canvas.canvasx(mouse_x)  # get coordinates of the mouse on the canvas
        y = self.canvas.canvasy(mouse_y)
        image_box = self.canvas.coords(self.container)
        rec_x0, rec_x1 = self.get_drawing_rectangle_x_coords(x, image_box)
        rec_y0, rec_y1 = self.get_drawing_rectangle_y_coords(y, image_box)
        return (rec_x0, rec_y0, rec_x1, rec_y1)

    def get_cropping_rectangle_coords(self, mouse_x, mouse_y):
        rec_x0, rec_y0, rec_x1, rec_y1 = self.get_drawing_rectangle_coords(mouse_x, mouse_y)
        rec_x0 = math.floor(rec_x0 / self.imscale)
        rec_y0 = math.floor(rec_y0 / self.imscale)
        rec_x1 = rec_x0 + 128
        rec_y1 = rec_y0 + 128
        return (rec_x0, rec_y0, rec_x1, rec_y1)

    def __draw_rectangle(self, event: tk.Event):
        if self.__drawed_rectangle:
            self.canvas.delete(self.__drawed_rectangle)

        self.__drawed_rectangle = self.canvas.create_rectangle(
            *self.get_drawing_rectangle_coords(event.x, event.y),
            outline='green'
        )

    def wheel(self, event: tk.Event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:
            scale        *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale        /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def set_image(self, image):
        self.image = image
        self.show_image()

    def show_image(self):
        ''' Show image on the Canvas '''
        if self.container:
            self.canvas.delete(self.container)
        if self.imageid:
            self.canvas.delete(self.imageid)
            self.imageid = None
            self.canvas.imagetk = None  # delete previous image from the canvas
        width, height = self.image.size
        new_size = int(self.imscale * width), int(self.imscale * height)
        self.resized_image = self.image.resize(new_size)
        imagetk = ImageTk.PhotoImage(self.resized_image)
        self.container = self.canvas.create_rectangle(
            (0, 0, self.resized_image.width, self.resized_image.height), width=0)
        self.imageid = self.canvas.create_image((0,0), anchor='nw', image=imagetk)
        self.canvas.lower(self.imageid)  # set it into background
        self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection
