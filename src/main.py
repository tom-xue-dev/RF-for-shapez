import sys
import subprocess
from pathlib import Path


def launch_processes():
    base_dir = Path(__file__).resolve().parent
    front_end = base_dir / "automation_script" / "front_end.py"
    game_exe = base_dir / "shapez" / "exe" / "exe" / "MyShapez.exe"

    if not front_end.exists():
        print(f"front_end.py not found: {front_end}")
        sys.exit(1)
    if not game_exe.exists():
        print(f"Game executable not found: {game_exe}")
        sys.exit(1)

    procs = []
    try:
        # Launch game executable in its own working directory (loads DLLs/resources correctly)
        print(f"Launching game: {game_exe}")
        game_proc = subprocess.Popen(
            [str(game_exe)],
            cwd=str(game_exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        procs.append(game_proc)

        # Launch front_end UI with Python in its script directory (ensures relative assets load)
        print(f"Launching front_end: {front_end}")
        front_proc = subprocess.Popen(
            [sys.executable, str(front_end)],
            cwd=str(front_end.parent),
        )
        procs.append(front_proc)

        # Wait for either to exit; keep main alive until both exit or interrupted
        while procs:
            for p in list(procs):
                ret = p.poll()
                if ret is not None:
                    # Process exited
                    print(f"Process exited (pid={p.pid}, code={ret})")
                    procs.remove(p)
            if procs:
                # Avoid busy loop
                try:
                    # Wait a bit on the first process
                    procs[0].wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass

    except KeyboardInterrupt:
        print("Interrupted. Terminating child processes...")
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    launch_processes()
