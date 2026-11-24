import numpy as np
def scale_lut(bits):
    hist = np.bincount((bits.ravel() & 32767), minlength=32768)
    return np.minimum(3, (2 * np.cumsum(hist) - hist) * 4 // (2 * bits.size)).astype(np.uint8)

