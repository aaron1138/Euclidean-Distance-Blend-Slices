from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QStackedWidget, QCheckBox, QSlider, QSpinBox,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
import numpy as np
import json

from config import Config, OrthogonalEngineConfig, GradientSlotMode, LutParameters
from gradient_visualizer import GradientVisualizer
import orthogonal_engine
import lut_manager

class Orthogonal1DTab(QWidget):
    """
    PySide6 tab for managing the Orthogonal 1D engine settings.
    """

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = parent_gui.config
        self._block_signals = False

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel_layout = QVBoxLayout()

        general_group = QGroupBox("General Settings")
        general_layout = QVBoxLayout(general_group)
        self.enable_curvature_checkbox = QCheckBox("Enable Dynamic Curvature Weighting")
        general_layout.addWidget(self.enable_curvature_checkbox)
        left_panel_layout.addWidget(general_group)

        gen_group = QGroupBox("Gradient Slot Generation")
        gen_layout = QVBoxLayout(gen_group)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Slot Generation Mode:"))
        self.slot_mode_combo = QComboBox()
        for mode in GradientSlotMode:
            self.slot_mode_combo.addItem(mode.value, mode)
        mode_layout.addWidget(self.slot_mode_combo)
        gen_layout.addLayout(mode_layout)

        self.params_stacked_widget = QStackedWidget()
        self._create_parameter_widgets()
        gen_layout.addWidget(self.params_stacked_widget)
        left_panel_layout.addWidget(gen_group)

        io_group = QGroupBox("Save/Load Full Gradient Set")
        io_layout = QHBoxLayout(io_group)
        self.save_slots_button = QPushButton("Save Gradient Set...")
        self.load_slots_button = QPushButton("Load Gradient Set...")
        io_layout.addWidget(self.save_slots_button)
        io_layout.addWidget(self.load_slots_button)
        left_panel_layout.addWidget(io_group)

        left_panel_layout.addStretch(1)
        main_layout.addLayout(left_panel_layout)

        right_panel_layout = QVBoxLayout()
        vis_group = QGroupBox("Gradient Preview")
        vis_layout = QVBoxLayout(vis_group)
        self.gradient_visualizer = GradientVisualizer(self)
        self.gradient_visualizer.setMinimumHeight(100)
        vis_layout.addWidget(self.gradient_visualizer)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Preview Run Length:"))
        self.preview_run_length_slider = QSlider(Qt.Horizontal)
        self.preview_run_length_slider.setRange(1, 255)
        selector_layout.addWidget(self.preview_run_length_slider)
        self.preview_run_length_spinbox = QSpinBox()
        self.preview_run_length_spinbox.setRange(1, 255)
        selector_layout.addWidget(self.preview_run_length_spinbox)
        vis_layout.addLayout(selector_layout)
        right_panel_layout.addWidget(vis_group)
        right_panel_layout.addStretch(1)
        main_layout.addLayout(right_panel_layout, 1)

    def _create_parameter_widgets(self):
        self.linear_params_widget = QWidget()
        linear_layout = QVBoxLayout(self.linear_params_widget); linear_layout.addWidget(QLabel("Generates a simple linear gradient for each run length.")); linear_layout.addStretch(1)
        self.params_stacked_widget.addWidget(self.linear_params_widget)

        self.equation_params_widget = QWidget()
        eq_layout = QVBoxLayout(self.equation_params_widget)
        eq_type_layout = QHBoxLayout(); eq_type_layout.addWidget(QLabel("Equation Type:"))
        self.equation_type_combo = QComboBox(); self.equation_type_combo.addItems(["gamma", "log", "exp"])
        eq_type_layout.addWidget(self.equation_type_combo); eq_layout.addLayout(eq_type_layout)
        self.eq_params_stacked_widget = QStackedWidget()

        self.gamma_param_widget = QWidget()
        gamma_l, self.gamma_edit, self.gamma_slider = self._create_slider_combo("Gamma:", (0.1, 10.0), 100)
        self.gamma_param_widget.setLayout(gamma_l)
        self.eq_params_stacked_widget.addWidget(self.gamma_param_widget)

        self.log_param_widget = QWidget()
        log_l, self.log_edit, self.log_slider = self._create_slider_combo("Param:", (0.1, 100.0), 10)
        self.log_param_widget.setLayout(log_l)
        self.eq_params_stacked_widget.addWidget(self.log_param_widget)

        self.exp_param_widget = QWidget()
        exp_l, self.exp_edit, self.exp_slider = self._create_slider_combo("Param:", (0.1, 10.0), 100)
        self.exp_param_widget.setLayout(exp_l)
        self.eq_params_stacked_widget.addWidget(self.exp_param_widget)

        eq_layout.addWidget(self.eq_params_stacked_widget)
        self.params_stacked_widget.addWidget(self.equation_params_widget)

        self.table_params_widget = QWidget()
        table_layout = QVBoxLayout(self.table_params_widget)
        table_path_layout = QHBoxLayout(); table_path_layout.addWidget(QLabel("Master LUT File:"))
        self.table_path_edit = QLineEdit(); self.table_path_edit.setReadOnly(True)
        table_path_layout.addWidget(self.table_path_edit)
        self.load_table_button = QPushButton("Load Master LUT..."); table_layout.addWidget(self.load_table_button)
        table_layout.addStretch(1)
        self.params_stacked_widget.addWidget(self.table_params_widget)

        self.piecewise_params_widget = QWidget()
        piecewise_layout = QVBoxLayout(self.piecewise_params_widget)
        piecewise_layout.addWidget(QLabel("Piecewise configuration is not yet available in the UI.")); piecewise_layout.addStretch(1)
        self.params_stacked_widget.addWidget(self.piecewise_params_widget)

        self.file_params_widget = QWidget()
        file_layout = QVBoxLayout(self.file_params_widget)
        file_path_layout = QHBoxLayout(); file_path_layout.addWidget(QLabel("Gradient Set File:"))
        self.gradient_set_path_edit = QLineEdit(); self.gradient_set_path_edit.setReadOnly(True)
        file_path_layout.addWidget(self.gradient_set_path_edit)
        file_layout.addLayout(file_path_layout)
        file_layout.addWidget(QLabel("To load a new set, use the 'Load Gradient Set...' button below."))
        file_layout.addStretch(1)
        self.params_stacked_widget.addWidget(self.file_params_widget)

    def _create_slider_combo(self, label, val_range, scale_factor):
        layout = QHBoxLayout(); layout.addWidget(QLabel(label))
        line_edit = QLineEdit(); line_edit.setFixedWidth(60)
        line_edit.setValidator(QDoubleValidator(val_range[0], val_range[1], 2, self))
        slider = QSlider(Qt.Horizontal); slider.setRange(int(val_range[0] * scale_factor), int(val_range[1] * scale_factor))
        layout.addWidget(line_edit); layout.addWidget(slider)
        return layout, line_edit, slider

    def _connect_signals(self):
        self.slot_mode_combo.currentIndexChanged.connect(self._on_slot_mode_changed)
        self.equation_type_combo.currentIndexChanged.connect(self._on_equation_type_changed)
        self.enable_curvature_checkbox.stateChanged.connect(self.save_settings)
        self.preview_run_length_slider.valueChanged.connect(self.preview_run_length_spinbox.setValue)
        self.preview_run_length_spinbox.valueChanged.connect(self.preview_run_length_slider.setValue)
        self.preview_run_length_spinbox.valueChanged.connect(self.update_visualization)
        self._connect_slider_combo_signals(self.gamma_edit, self.gamma_slider, "gamma_value", 100.0)
        self._connect_slider_combo_signals(self.log_edit, self.log_slider, "log_param", 10.0)
        self._connect_slider_combo_signals(self.exp_edit, self.exp_slider, "exp_param", 100.0)
        self.load_table_button.clicked.connect(self._load_master_lut_from_file)
        self.save_slots_button.clicked.connect(self._save_gradient_set_to_file)
        self.load_slots_button.clicked.connect(self._load_gradient_set_from_file)

    def _connect_slider_combo_signals(self, line_edit, slider, param_name, scale_factor):
        slider.valueChanged.connect(lambda val, le=line_edit: le.setText(f"{val / scale_factor:.2f}"))
        slider.sliderReleased.connect(self.update_visualization)
        line_edit.editingFinished.connect(self.update_visualization)

    def _on_slot_mode_changed(self, index):
        self.params_stacked_widget.setCurrentIndex(index)
        self.update_visualization()
        self.save_settings()

    def _on_equation_type_changed(self):
        self.eq_params_stacked_widget.setCurrentIndex(self.equation_type_combo.currentIndex())
        self.update_visualization()

    def update_visualization(self):
        if self._block_signals: return
        self.save_settings()
        run_length = self.preview_run_length_spinbox.value()
        temp_slots = orthogonal_engine.precompute_gradient_slots(self.config.orthogonal_engine)
        if temp_slots and len(temp_slots) > run_length:
            self.gradient_visualizer.update_gradient(temp_slots[run_length])

    def _load_master_lut_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Master LUT File", "", "JSON Files (*.json)")
        if filepath:
            try:
                lut_manager.load_lut(filepath)
                self.config.orthogonal_engine.gradient_table_path = filepath
                self.apply_settings(self.config)
                self.update_visualization()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT file: {e}")

    def _save_gradient_set_to_file(self):
        self.save_settings()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Gradient Set", "2D-fade-LUT.json", "JSON Files (*.json)")
        if filepath:
            try:
                slots_to_save = orthogonal_engine.precompute_gradient_slots(self.config.orthogonal_engine)
                serializable_slots = [arr.tolist() for arr in slots_to_save]
                with open(filepath, 'w') as f:
                    json.dump(serializable_slots, f, indent=2)
                QMessageBox.information(self, "Save Success", "Gradient set saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save gradient set: {e}")

    def _load_gradient_set_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Gradient Set", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
                self.config.orthogonal_engine.gradient_slot_mode = GradientSlotMode.FILE
                self.config.orthogonal_engine.gradient_set_path = filepath
                self.apply_settings(self.config)
                QMessageBox.information(self, "Load Success", "Gradient set loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load gradient set: {e}")

    def apply_settings(self, config: Config):
        self._block_signals = True
        self.config = config
        oc = self.config.orthogonal_engine
        eq_params = oc.gradient_equation_params

        self.enable_curvature_checkbox.setChecked(oc.enable_dynamic_curvature)
        self.slot_mode_combo.setCurrentIndex(self.slot_mode_combo.findData(oc.gradient_slot_mode))
        self.params_stacked_widget.setCurrentIndex(self.slot_mode_combo.currentIndex())
        self.table_path_edit.setText(oc.gradient_table_path)
        self.gradient_set_path_edit.setText(oc.gradient_set_path)

        self.equation_type_combo.setCurrentText(eq_params.lut_generation_type)
        self.eq_params_stacked_widget.setCurrentIndex(self.equation_type_combo.currentIndex())

        self.gamma_edit.setText(f"{eq_params.gamma_value:.2f}")
        self.gamma_slider.setValue(int(eq_params.gamma_value * 100))
        self.log_edit.setText(f"{eq_params.log_param:.1f}")
        self.log_slider.setValue(int(eq_params.log_param * 10))
        self.exp_edit.setText(f"{eq_params.exp_param:.2f}")
        self.exp_slider.setValue(int(eq_params.exp_param * 100))

        self._block_signals = False
        self.update_visualization()

    def save_settings(self):
        if self._block_signals: return
        oc = self.config.orthogonal_engine
        eq_params = oc.gradient_equation_params

        oc.enable_dynamic_curvature = self.enable_curvature_checkbox.isChecked()
        oc.gradient_slot_mode = self.slot_mode_combo.currentData()
        oc.gradient_table_path = self.table_path_edit.text()
        oc.gradient_set_path = self.gradient_set_path_edit.text()

        eq_params.lut_generation_type = self.equation_type_combo.currentText()
        try: eq_params.gamma_value = float(self.gamma_edit.text())
        except (ValueError, TypeError): pass
        try: eq_params.log_param = float(self.log_edit.text())
        except (ValueError, TypeError): pass
        try: eq_params.exp_param = float(self.exp_edit.text())
        except (ValueError, TypeError): pass
