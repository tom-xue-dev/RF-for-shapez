# RF-for-shapez

Reinforcement learning for the game Shapez using a PPO agent to automatically plan and place structures (conveyors, miner, cutter, trash, etc.). The project reads map/goal information from the game via shared memory, provides a PyQt5 frontend to select models and run inference or training, and uses PyAutoGUI to perform actions in the game window.

Highlights:
- Read game state from shared memory (resource grid, building grid, target shape, scroll offset, scale factor).
- Load or train PPO models (Stable-Baselines3) with an action mask to restrict invalid actions.
- Frontend UI to choose `.zip` models, set timesteps/hyperparameters, and start/pause/stop execution.
- Backend automates mouse/keyboard in the SHAPEZ window via screen matching and input simulation.

Key files:
- `src/main.py`: Launches the game executable and the frontend UI together.
- `src/automation_script/front_end.py`: PyQt5 UI.
- `src/automation_script/back_end.py`: Screen operations and placement logic (PyAutoGUI).
- `src/ppo_model/PPO.py`: PPO load/train logic with a masked policy.
- `src/ppo_model/ShapezEnv.py`: Custom Gymnasium environment.
- `src/ppo_model/getmap.py`: Read map/goal/scroll/scale from shared memory.
- `src/shapez/exe/exe/MyShapez.exe`: Game executable (you must provide it; see below).
- `src/ppo_model/trained/*.zip`: Example trained models.

Note: Windows only (uses WinAPI window queries, VC Runtime, PyAutoGUI, and Windows shared memory naming).

## Environment Setup

Conda (or Miniconda) recommended:
- Download: https://www.anaconda.com/products/distribution

Option A: Use provided `environment.yml` (recommended)
- From the repo root:
  ```bash
  conda env create -f environment.yml
  conda activate rf_shapez
  ```
  Tip: `environment.yml` was exported on Windows and may be UTF-16 encoded. If Conda fails to parse it, re-save as UTF-8 and retry.

Option B: Minimal manual setup (if you don’t want `environment.yml`)
- Create Python 3.9 env and install core deps:
  ```bash
  conda create -n rf_shapez python=3.9 -y
  conda activate rf_shapez
  pip install gymnasium numpy pyqt5 pyautogui pygetwindow pyscreeze opencv-python pillow \
             stable-baselines3==2.0.0 torch torchvision torchaudio
  ```
  - For GPU, use the official PyTorch instructions matching your CUDA.
  - `requirements.txt` contains Windows-exported, machine-specific wheels/paths and may not be portable across machines.

## Prerequisites (Important)

- Game executable: place a Shapez build that exposes the required shared memory at `src/shapez/exe/exe/MyShapez.exe`.
  - If you keep it elsewhere, update `game_exe` in `src/main.py` accordingly.
  - The window title must be `SHAPEZ` (used by `visualize.py` to find the window).
- Windowing: run in windowed mode. A visible area around `1200x830` matches defaults in `back_end.py`.
- Shared memory segments expected by this project:
  - `SharedResource`: int32 grid, 20x32 (resource map)
  - `SharedBuild`: int32 grid, 20x32 (buildings map)
  - `need_shape_name`: int32, shape id (1,)
  - `scrolloffset`: int32, (2,) scroll offset
  - `scaleFactor`: float64, (1,) UI scale factor
- Toolbar icons for screen matching: `src/automation_script/*.png` must match your game UI theme/scale; otherwise detection may fail.

## Run & Inference

Option 1: One-command launch (recommended)
- From repo root:
  ```bash
  conda activate rf_shapez
  python src/main.py
  ```
  This starts `MyShapez.exe` (in its own folder so DLL/resources load properly) and then launches the PyQt5 UI.

Option 2: UI only (if you manually start the game)
- From repo root:
  ```bash
  conda activate rf_shapez
  python src/automation_script/front_end.py
  ```

In the UI:
- Click “Choose Model” and select a `.zip` SB3 model (e.g., from `src/ppo_model/trained`).
- “Enter max steps” is currently not used by the inference path (kept for forward compatibility).
- Click “Start” to run placement; “Pause” toggles pause; “Stop” clears output and resets selection.

Note: If no model is selected, the code attempts to load `src/ppo_model/ppo_model.zip`. If it doesn’t exist, loading will fail. Please choose a `.zip` or train one first.

## Training

The UI has a training panel:
- `Train timesteps`: number of timesteps (e.g., `100000`).
- `Gamma`: discount factor (default `0.98`).
- `Save directory`: optional output directory (defaults to `src/ppo_model/trained`).
- `Model name`: optional filename; `.zip` is appended automatically.

Click “Train Model” to start. The trainer:
- Builds a `ShapezEnv` from the current shared memory state and target shape.
- Trains using `MaskedMultiInputPolicy` and `ActionMaskCallback` to enforce valid actions.
- Saves a `.zip` SB3 model to the chosen directory.

You can also call `ppo_model.PPO.train_model` from code if you prefer scripting.

## FAQ

- Game executable not found
  - When running `src/main.py`, if `MyShapez.exe` is missing, place it under `src/shapez/exe/exe/` or change the path in `src/main.py`.
- Shared memory missing or wrong size
  - Errors like “SharedResource not found/size < expected” indicate the game is not providing the expected segments or shapes/dtypes (20x32 int32). Start the correct build and try again.
- Toolbar icon detection fails
  - `back_end.py` uses `belt.png`/`miner.png`/`cutter.png`/`trash.png` to click toolbar buttons. Ensure your UI theme/language/scale matches those PNGs; replace them if needed.
- Model fails to load
  - No model selected or default `ppo_model.zip` missing; or SB3/Torch version mismatch. Train a new model with the current environment and try again.
- Windows only
  - Relies on WinAPI window discovery and desktop automation; other platforms are not supported.

## Development Notes

- Change game path: edit `game_exe` in `src/main.py`.
- Adjust window/grid params: see `Width/Height/Horizontal/verticle` and `cell_size` calculation in `src/automation_script/back_end.py` (affected by `scaleFactor`).
- If slimming deps, keep at least: `gymnasium`, `numpy`, `pyqt5`, `pyautogui`, `stable-baselines3`, `torch`, `opencv-python`, `pillow`.


