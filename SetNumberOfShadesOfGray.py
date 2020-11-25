import tkinter as tk

from typing import List

class SetNumberOfShadesOfGray:
    def __init__(self, parent: tk.Tk) -> None:
        self.parent = parent
        self.root = tk.Toplevel(self.parent)
        self.root.title("Set number of shades of gray")
        self.root.geometry("200x50")

        self.numberOfShades = tk.StringVar(value='32')
        self.numberOfShadesEntry = tk.Entry(self.root, textvariable=self.numberOfShades)
        self.numberOfShadesEntry.pack()
