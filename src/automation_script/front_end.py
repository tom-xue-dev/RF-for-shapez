import sys
import os
import importlib.util
import traceback
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QTextEdit, QPushButton, QLineEdit, QFileDialog, QScrollArea, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QWaitCondition, QMutex
import re  # For sanitizing the file name


class FileExecutorThread(QThread):
    output_signal = pyqtSignal(str)
    pause_signal = pyqtSignal(bool)

    def __init__(self, file_path, steps, model_path=None):
        super().__init__()
        self.file_path = file_path
        self.steps = steps
        self.model_path = model_path
        self._pause = False
        self._pause_cond = QWaitCondition()
        self._mutex = QMutex()

    def run(self):
        try:
            spec = importlib.util.spec_from_file_location("module.name", self.file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Prefer a 'run(model_path=None, steps=None)' entry
            if hasattr(module, 'run'):
                self.output_signal.emit(f"Running {self.file_path} with model: {self.model_path or '(default)'}...")
                result = module.run(self.model_path, self.steps)
                if result is not None:
                    self.output_signal.emit(str(result))
            elif hasattr(module, 'main'):
                self.output_signal.emit(f"Running main() in {self.file_path} ...")
                module.main()
            else:
                self.output_signal.emit(f"Executed {self.file_path}, but no 'run' or 'main' function found.")
        except Exception as e:
            error_message = f"Error during execution: {traceback.format_exc()}"
            self.output_signal.emit(error_message)

    def toggle_pause(self):
        """Pause or resume execution."""
        self._mutex.lock()
        self._pause = not self._pause
        if not self._pause:
            self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.pause_signal.emit(self._pause)


from typing import Optional

class TrainingThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, timesteps: int, save_dir: Optional[str], model_name: Optional[str], gamma: Optional[float]):
        super().__init__()
        self.timesteps = timesteps
        self.save_dir = save_dir
        self.model_name = model_name
        self.gamma = gamma

    def run(self):
        try:
            from ppo_model.PPO import train_model
            try:
                import torch as th
                device_hint = "cuda" if th.cuda.is_available() else "cpu"
            except Exception:
                device_hint = "cpu"
            self.output_signal.emit(
                f"Training started (timesteps={self.timesteps}, gamma={self.gamma or 0.98}, device={device_hint}, save_dir={self.save_dir or '(default)'}, model_name={self.model_name or '(auto)'})."
            )
            out_path = train_model(
                total_timesteps=int(self.timesteps) if self.timesteps else 10000,
                save_dir=self.save_dir if self.save_dir else None,
                model_name=self.model_name if self.model_name else None,
                gamma=float(self.gamma) if self.gamma else 0.98,
                device=device_hint,
            )
            self.output_signal.emit(f"Training finished. Model saved to: {out_path}")
            self.finished_signal.emit(out_path)
        except Exception:
            self.output_signal.emit(f"Training error: {traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Python File Executor")
        self.setFixedSize(900, 600)

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Initialize scroll area but don't show buttons until a file is selected
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedSize(300, 600)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        # Create initial "Choose Model" button
        self.choose_model_button = QPushButton("Choose Model")
        self.choose_model_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.choose_model_button.clicked.connect(self.choose_model)

        # Add the "Choose Model" button to the layout
        self.scroll_layout.addWidget(self.choose_model_button)

        main_layout.addWidget(self.scroll_area)

        # Right side layout with text area and input steps
        right_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setFixedSize(500, 300)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(self.text_edit)

        self.input_steps = QLineEdit(self)
        self.input_steps.setPlaceholderText("Enter max steps")
        right_layout.addWidget(self.input_steps)

        button_layout = QHBoxLayout()

        # Start button
        start_button = QPushButton("Start")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        start_button.clicked.connect(self.start_execution)
        button_layout.addWidget(start_button)

        # Pause/Resume button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #ec971f;
            }
        """)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause_execution)
        button_layout.addWidget(self.pause_button)

        # Stop button
        stop_button = QPushButton("Stop")
        stop_button.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
        """)
        stop_button.clicked.connect(self.stop_execution)
        button_layout.addWidget(stop_button)

        right_layout.addLayout(button_layout)

        # ---- Training controls ----
        self.train_timesteps = QLineEdit(self)
        self.train_timesteps.setPlaceholderText("Train timesteps (e.g. 100000)")
        right_layout.addWidget(self.train_timesteps)

        gamma_row = QHBoxLayout()
        self.train_gamma = QLineEdit(self)
        self.train_gamma.setPlaceholderText("Gamma (default 0.98)")
        gamma_row.addWidget(self.train_gamma)
        right_layout.addLayout(gamma_row)

        save_dir_row = QHBoxLayout()
        self.train_save_dir = QLineEdit(self)
        self.train_save_dir.setPlaceholderText("Save directory (optional)")
        save_dir_row.addWidget(self.train_save_dir)
        browse_save_dir_btn = QPushButton("Browse Dir")
        browse_save_dir_btn.clicked.connect(self.browse_save_dir)
        save_dir_row.addWidget(browse_save_dir_btn)
        right_layout.addLayout(save_dir_row)

        self.train_model_name = QLineEdit(self)
        self.train_model_name.setPlaceholderText("Model name (optional, .zip appended)")
        right_layout.addWidget(self.train_model_name)

        train_btn = QPushButton("Train Model")
        train_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        train_btn.clicked.connect(self.start_training)
        right_layout.addWidget(train_btn)
        main_layout.addLayout(right_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.executor_thread = None
        self.training_thread = None
        # Auto-select back_end.py as the script to run
        self.selected_python_file = str((Path(__file__).parent / 'back_end.py').resolve())
        self.selected_model_file = None  # Track currently selected model (.zip)
        self.file_buttons = {}  # Dictionary to store file paths and associated buttons

        # Inform user
        self.text_edit.append(f"Auto-selected script: {self.selected_python_file}")

    def sanitize_variable_name(self, file_name):
        """Sanitize the file name to create a valid Python variable name."""
        sanitized_name = re.sub(r'\W|^(?=\d)', '_', file_name)  # Replace non-alphanumeric characters with underscores
        return sanitized_name

    def choose_model(self):
        # Open file dialog to select a PPO model file (.zip)
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose PPO Model",
            "",
            "Zip Files (*.zip)",
            options=options,
        )
        if file_name:
            self.add_file_button(file_name)


    def add_file_button(self, file_path):
        """Add a button for the selected model (.zip) and store its path."""
        # Validate extension
        if not file_path.lower().endswith(".zip"):
            self.text_edit.append("Please choose a PPO model file (.zip).")
            return
        file_name = os.path.basename(file_path)
        variable_name = self.sanitize_variable_name(os.path.splitext(file_name)[0])  # Remove extension and sanitize

        # Check if the file is already loaded
        if variable_name in globals():
            self.text_edit.append(f"Model {file_name} is already loaded as variable {variable_name}.")
            return

        # Create a global variable with the sanitized name
        globals()[variable_name] = file_path
        self.text_edit.append(f"Created model variable '{variable_name}': {file_path}")

        # Create a layout for each variable with its name, input, and buttons
        var_layout = QVBoxLayout()

        # Label for showing variable name and current value
        label = QLabel(f"Variable: {variable_name}")
        var_layout.addWidget(label)

        # Input field to modify the variable (file path)
        var_input = QLineEdit(self)
        var_input.setText(file_path)
        var_layout.addWidget(var_input)

        # Update button to modify the variable's value
        update_button = QPushButton("Update Variable")
        update_button.clicked.connect(lambda: self.update_variable(variable_name, var_input.text()))
        var_layout.addWidget(update_button)

        # Use button to mark this as the active model
        run_button = QPushButton(f"Use model: {variable_name}")
        run_button.clicked.connect(lambda: self.model_clicked(file_path))
        var_layout.addWidget(run_button)

        # Clear button to remove the variable
        clear_button = QPushButton(f"Clear {variable_name}")
        clear_button.clicked.connect(lambda: self.clear_variable(variable_name, var_layout))
        var_layout.addWidget(clear_button)

        # Add the variable layout to the scroll area
        self.scroll_layout.addLayout(var_layout)
        self.file_buttons[file_name] = file_path

    def update_variable(self, var_name, new_value):
        """Update the value of the variable and display in the text edit."""
        globals()[var_name] = new_value
        self.text_edit.append(f"Updated variable '{var_name}' to new value: {new_value}")

    def clear_variable(self, var_name, layout):
        """Clear the variable and remove its layout from the UI."""
        if var_name in globals():
            del globals()[var_name]
            self.text_edit.append(f"Cleared variable '{var_name}'")

        # Hide the layout from the UI
        for i in reversed(range(layout.count())): 
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def model_clicked(self, file_path):
        """Handle clicking a model variable and set selected model file."""
        self.text_edit.append(f"Selected model: {file_path}")
        self.selected_model_file = file_path

    def start_execution(self):
        """Start execution of the selected Python file."""
        if self.selected_python_file:
            steps = self.input_steps.text()
            if steps.isdigit():
                steps = int(steps)
            else:
                steps = None

            self.text_edit.append(
                f"Executing script: {self.selected_python_file}\nUsing model: {self.selected_model_file or '(default)'}\n"
            )

            self.executor_thread = FileExecutorThread(self.selected_python_file, steps, model_path=self.selected_model_file)
            self.executor_thread.output_signal.connect(self.append_output)
            self.executor_thread.start()

            # Enable pause button after starting
            self.pause_button.setEnabled(True)
        else:
            self.text_edit.append("No valid script configured.")

    def append_output(self, output):
        """Display output in the text edit."""
        self.text_edit.append(output)

    def toggle_pause_execution(self):
        """Toggle pause/resume in the thread."""
        if self.executor_thread:
            self.executor_thread.toggle_pause()

            # Update pause button text based on the state
            if self.executor_thread._pause:
                self.pause_button.setText("Resume")
            else:
                self.pause_button.setText("Pause")

    def stop_execution(self):
        """Stop the current execution, clear the output, and reset the selected file."""
        # Clear the output area
        self.text_edit.clear()
        self.text_edit.append("Output cleared.")

        # Reset the selected Python file
        self.selected_python_file = None
        self.text_edit.append("Selected file has been cleared.")

        # Reset pause button
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)  # Disable the pause button when stopped

    def browse_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Save Directory", "")
        if directory:
            self.train_save_dir.setText(directory)

    def start_training(self):
        try:
            timesteps_text = self.train_timesteps.text().strip()
            timesteps = int(timesteps_text) if timesteps_text.isdigit() else 10000
        except Exception:
            timesteps = 10000

        gamma_text = self.train_gamma.text().strip()
        try:
            gamma = float(gamma_text) if gamma_text else 0.98
        except Exception:
            gamma = 0.98

        save_dir = self.train_save_dir.text().strip() or None
        model_name = self.train_model_name.text().strip() or None

        self.text_edit.append(
            f"Start training with timesteps={timesteps}, gamma={gamma}, save_dir={save_dir or '(default)'}, model_name={model_name or '(auto)'}"
        )

        self.training_thread = TrainingThread(timesteps, save_dir, model_name, gamma)
        self.training_thread.output_signal.connect(self.append_output)
        self.training_thread.finished_signal.connect(lambda p: self.text_edit.append(f"Model ready: {p}"))
        self.training_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
