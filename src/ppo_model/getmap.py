import glob
from multiprocessing import shared_memory
import numpy as np
import time

HEIGHT = 20
WIDTH = 32
INT_DTYPE = np.int32


def _ensure_size(buf, expected_bytes, name: str):
    actual = len(buf)
    if actual < expected_bytes:
        raise RuntimeError(
            f"Shared memory '{name}' size {actual}B < expected {expected_bytes}B. "
            f"Game not ready or dtype/shape mismatch (expected {HEIGHT}x{WIDTH} {INT_DTYPE})."
        )


def load_shared_arrays(retries: int = 10, interval: float = 0.2):
    res_shm = bld_shm = None
    try:
        # Retry a few times to wait for game to initialize shared memory
        for _ in range(retries):
            try:
                res_shm = shared_memory.SharedMemory(name='SharedResource')
                bld_shm = shared_memory.SharedMemory(name='SharedBuild')
                break
            except FileNotFoundError:
                time.sleep(interval)
        if res_shm is None or bld_shm is None:
            raise FileNotFoundError("SharedResource/SharedBuild not found. Ensure game is running.")

        expected = HEIGHT * WIDTH * np.dtype(INT_DTYPE).itemsize
        _ensure_size(res_shm.buf, expected, 'SharedResource')
        _ensure_size(bld_shm.buf, expected, 'SharedBuild')

        resource_array = np.ndarray((HEIGHT, WIDTH), dtype=INT_DTYPE, buffer=res_shm.buf).copy()
        buildingsmap_array = np.ndarray((HEIGHT, WIDTH), dtype=INT_DTYPE, buffer=bld_shm.buf).copy()

        belt_mapping = {
            31: 3103, 32: 3111, 33: 3109, 34: 3104, 35: 3112, 36: 3110,
            37: 3102, 38: 3107, 39: 3108, 40: 3101, 41: 3105, 42: 3106,
        }
        for i in range(buildingsmap_array.shape[0]):
            for j in range(buildingsmap_array.shape[1]):
                item = buildingsmap_array[i, j]
                if item in belt_mapping:
                    buildingsmap_array[i, j] = belt_mapping[item]

        return resource_array, buildingsmap_array
    finally:
        try:
            if res_shm is not None:
                res_shm.close()
        except Exception:
            pass
        try:
            if bld_shm is not None:
                bld_shm.close()
        except Exception:
            pass


def load_needed_shape():
    shm = None
    try:
        shm = shared_memory.SharedMemory(name='need_shape_name')
        expected = 1 * np.dtype(INT_DTYPE).itemsize
        _ensure_size(shm.buf, expected, 'need_shape_name')
        shape = np.ndarray((1,), dtype=INT_DTYPE, buffer=shm.buf)[0]
        return int(shape)
    finally:
        try:
            if shm is not None:
                shm.close()
        except Exception:
            pass


def load_scroll_offset():
    shm = None
    try:
        shm = shared_memory.SharedMemory(name='scrolloffset')
        expected = 2 * np.dtype(INT_DTYPE).itemsize
        _ensure_size(shm.buf, expected, 'scrolloffset')
        arr = np.ndarray((2,), dtype=INT_DTYPE, buffer=shm.buf)
        return int(arr[0]), int(arr[1])
    finally:
        try:
            if shm is not None:
                shm.close()
        except Exception:
            pass


def load_scaleFactor():
    shm = None
    try:
        shm = shared_memory.SharedMemory(name='scaleFactor')
        expected = np.dtype(np.double).itemsize
        _ensure_size(shm.buf, expected, 'scaleFactor')
        scaleFactor = np.frombuffer(shm.buf, dtype=np.double)[0]
        return float(scaleFactor)
    finally:
        try:
            if shm is not None:
                shm.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Simple sanity check when run directly
    res, bld = load_shared_arrays()
    print("resource shape:", res.shape, res.dtype)
    print("build shape:", bld.shape, bld.dtype)
    print("scroll:", load_scroll_offset())
    print("scale:", load_scaleFactor())
