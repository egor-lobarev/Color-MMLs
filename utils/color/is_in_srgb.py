import numpy as np
from colour import xyY_to_XYZ, XYZ_to_sRGB

def is_in_srgb_gamut(xyY: list[float], reference_white_Y: float=100.0) -> bool:
    """
    Check if given xyY color is within the sRGB gamut by converting to linear sRGB
    (no encoding) and verifying all channels are in [0, 1] without clipping.
    """
    try:
        xyY = np.array(xyY, dtype=float)
        XYZ = xyY_to_XYZ(xyY)
        # normalize to reference white luminance
        XYZ = XYZ / reference_white_Y
        # Linear sRGB
        rgb_linear = XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
        return np.all(rgb_linear >= 0.0) and np.all(rgb_linear <= 1.0)
    except Exception:
        return False