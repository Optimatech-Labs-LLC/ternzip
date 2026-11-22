"""Optional high-CPU wait for the packaged 9B greedy Metal decoder."""
import ctypes as C


def enable_fast_wait(engine, backend_dir):
    if len(engine.devices) != 1 or not engine.sampler or not getattr(engine, "used_direct_writer", False):
        raise ValueError("Fast waiting requires the single-device direct 9B Metal loader.")
    # Obtain the already-registered MTL0 device. GGML owns it and its queue.
    metal = C.CDLL(str(backend_dir / "libggml-metal.0.dylib"))
    ggml = C.CDLL(str(backend_dir / "libggml.0.dylib"))
    device = ggml.ggml_backend_dev_by_name
    device.argtypes = [C.c_char_p]
    device.restype = C.c_void_p
    get_queue = metal.ggml_metal_device_get_queue
    get_queue.argtypes = [C.c_void_p]
    get_queue.restype = C.c_void_p
    C.CDLL("/usr/lib/libobjc.A.dylib", mode=C.RTLD_GLOBAL)
    companion = C.CDLL(str(backend_dir / "libternzip-wait.dylib"))
    registered = companion.ternzip_registered_device_context
    registered.argtypes = [C.c_void_p]
    registered.restype = C.c_void_p
    dev = registered(device(b"MTL0"))
    queue = get_queue(dev) if dev else None
    if not queue:
        raise RuntimeError("The Metal backend did not provide its command queue.")
    wait = companion.ternzip_wait_for_queue
    wait.argtypes = [C.c_void_p]
    wait.restype = None
    engine.wait_for_generation = lambda: wait(queue)
