from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue 
import numpy as np

from PyQt5 import uic

class MplCanvas(FigureCanvas):
    def __init__(self,parent=None,width = 5, height = 4, dpi = 100):
        fig = Figure(figsize=(width,height),dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas,self).__init__(fig)
        fig.tight_layout()

class Py