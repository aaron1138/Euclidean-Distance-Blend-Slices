"""
Copyright (c) 2025 Aaron Baca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class GradientVisualizer(QWidget):
    """
    A widget to visualize a gradient as a horizontal bar chart.
    """
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_gradient(self, gradient_array: np.ndarray):
        """
        Updates the bar chart with a new gradient array.
        """
        self.axes.clear()
        if gradient_array is not None and len(gradient_array) > 0:
            # Create colors from the gradient array (0-255 -> 0.0-1.0)
            colors = [(g/255., g/255., g/255.) for g in gradient_array]
            self.axes.bar(range(len(gradient_array)), [1] * len(gradient_array), color=colors, width=1.0)

        self.axes.set_yticks([])
        self.axes.set_xlabel("Pixel")
        self.axes.set_title("Gradient Preview")
        self.figure.tight_layout()
        self.canvas.draw()
