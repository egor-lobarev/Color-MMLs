import random
from typing import List, Tuple, Optional

import colour
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class MunsellColor:
    def __init__(self, h: float, c: int, v: int):
        self.h = h  # hue
        self.c = c  # chroma
        self.v = v  # value

    def __repr__(self):
        return f"H={self.h}, C={self.c}, V={self.v}"

    @staticmethod
    def single(h, c, v) -> "MunsellColor":
        """
        Create a single Munsell color with parameters H, C, V.
        H can be a number (e.g., 5.0) or a string (e.g., '2.5R').
        C and V are integers.
        """
        return MunsellColor(h=h, c=int(c), v=int(v))

    @staticmethod
    def chain(
        variable: str,
        values: List,
        fixed_h: Optional[int] = None,
        fixed_c: Optional[int] = None,
        fixed_v: Optional[int] = None,
    ) -> List["MunsellColor"]:
        """
        Generate a chain of Munsell colors where two parameters are fixed
        and the third varies according to the provided values.

        :param variable: Which parameter to vary: 'h', 'c', or 'v'.
        :param values: Values to assign to the varying parameter.
        :param fixed_h: Fixed H (float|str) when variable != 'h'.
        :param fixed_c: Fixed C (int) when variable != 'c'.
        :param fixed_v: Fixed V (int) when variable != 'v'.
        :return: List of MunsellColor.
        """
        variable = variable.lower()
        if variable not in {"h", "c", "v"}:
            raise ValueError("variable must be one of {'h','c','v'}")

        # Ensure exactly two parameters are fixed
        fixed_count = sum(x is not None for x in (fixed_h, fixed_c, fixed_v))
        if fixed_count != 2:
            raise ValueError(
                "Exactly two of fixed_h, fixed_c, fixed_v must be provided."
            )

        # Validate fixed types to avoid accidental mutation of fixed params
        def _is_valid_hue(hval) -> bool:
            if isinstance(hval, str):
                # Rough validation for patterns like '2.5R', '10GY', 'N'
                # Accepts neutral 'N' and hue tokens ending with one of standard sectors
                import re

                return bool(
                    re.fullmatch(
                        r"N|((?:2\.5|5|7\.5|10)(R|YR|Y|GY|G|BG|B|PB|P|RP))", hval
                    )
                )
            # Allow numeric hue as fixed only if varying H; keep as is otherwise
            return isinstance(hval, (int, float))

        if variable != "h":
            if fixed_h is None:
                raise ValueError("fixed_h must be provided when variable!='h'")
            if not _is_valid_hue(fixed_h):
                raise ValueError(
                    "fixed_h must be a valid Munsell hue string like '2.5R', '5Y', '10PB', or 'N'"
                )

        if variable != "c":
            if fixed_c is None:
                raise ValueError("fixed_c must be provided when variable!='c'")
            fixed_c = int(fixed_c)

        if variable != "v":
            if fixed_v is None:
                raise ValueError("fixed_v must be provided when variable!='v'")
            fixed_v = int(fixed_v)

        chain: List[MunsellColor] = []
        for val in values:
            if variable == "h":
                # varying hue: val must be a valid hue string or numeric hue token
                h_val = val
                if not _is_valid_hue(h_val):
                    raise ValueError(
                        "values for variable='h' must be valid hue tokens like '2.5R', '7.5GY', 'N'"
                    )
                chain.append(MunsellColor(h=h_val, c=int(fixed_c), v=int(fixed_v)))
            elif variable == "c":
                # varying chroma: only C changes, keep H exactly as provided
                c_val = int(val)
                chain.append(MunsellColor(h=fixed_h, c=c_val, v=int(fixed_v)))
            else:  # variable == 'v'
                # varying value: only V changes, keep H and C exactly as provided
                v_val = int(val)
                chain.append(MunsellColor(h=fixed_h, c=int(fixed_c), v=v_val))

        return chain

    # Convenience explicit chain builders to avoid misuse
    @staticmethod
    def chain_vary_h(hues: List[str], c: int, v: int) -> List["MunsellColor"]:
        """Generate a chain varying H; C and V are fixed."""
        return MunsellColor.chain("h", hues, fixed_c=c, fixed_v=v)

    @staticmethod
    def chain_vary_c(h: str, chromas: List[int], v: int) -> List["MunsellColor"]:
        """Generate a chain varying C; H and V are fixed."""
        return MunsellColor.chain("c", chromas, fixed_h=h, fixed_v=v)

    @staticmethod
    def chain_vary_v(h: str, c: int, values_v: List[int]) -> List["MunsellColor"]:
        """Generate a chain varying V; H and C are fixed."""
        return MunsellColor.chain("v", values_v, fixed_h=h, fixed_c=c)


def generate_control_pairs(
    colors: List[MunsellColor], distance: float = 2.5
) -> List[Tuple[MunsellColor, MunsellColor]]:
    """Генерируем контрольные пары с большим воспринимаемым различием."""
    control_pairs = []
    for col in colors:
        # Случайно выбираем пару цветов из другого диапазона оттенков
        other_hue = random.choice(
            [val for val in range(0, 10) if abs(val - col.h) > distance]
        )
        new_col = MunsellColor(h=other_hue, c=col.c, v=col.v)
        control_pairs.append((col, new_col))
    return control_pairs



def _xyY_to_srgb_clipped(
    xyY: np.ndarray, reference_white_Y: float = 1.0
) -> np.ndarray:
    """
    Convert xyY to sRGB in [0, 1], normalizing Y by the given reference white if needed.

    Notes on normalization:
    - Many Munsell datasets express Y relative to a white of Y≈100.
    - If xyY[2] (Y) is > 1.5, we assume it's in 0–100 units and divide by reference_white_Y.
    - If it's already in 0–1, we leave it unchanged.
    """
    xyY = np.array(xyY, dtype=float)
    xyY[2] = xyY[2] / float(reference_white_Y)

    XYZ = colour.xyY_to_XYZ(xyY)
    srgb = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=True)
    return np.clip(srgb, 0.0, 1.0)


def munsell_swatch(
    munsell_specification: str,
    size: Tuple[int, int] = (200, 200),
    reference_white_Y: float = 1.0,
    return_numpy: bool = False,
):
    """
    Create a solid-colour swatch image for a given Munsell colour.

    Parameters:
    - munsell_specification: e.g., '5R 5/10' or 'N 5/'.
    - size: (width, height) in pixels.
    - reference_white_Y: Y value used to normalize luminance if the input Y is in 0–100.
    - return_numpy: if True, return a NumPy array in uint8; otherwise return a PIL Image.

    Returns:
    - PIL.Image.Image (RGB) by default, or NumPy uint8 array if return_numpy=True.
    """
    # Get xyY from Munsell notation
    xyY = colour.munsell_colour_to_xyY(munsell_specification)
    srgb = _xyY_to_srgb_clipped(xyY, reference_white_Y=reference_white_Y)

    # Build image buffer
    w, h = int(size[0]), int(size[1])
    rgb8 = (np.clip(srgb, 0, 1) * 255.0).round().astype(np.uint8)
    img_arr = np.tile(rgb8.reshape(1, 1, 3), (h, w, 1))

    if return_numpy:
        return img_arr
    return Image.fromarray(img_arr, mode="RGB")


def munsell_csv_row_swatch(
    x: float,
    y: float,
    Y: float,
    size: Tuple[int, int] = (200, 200),
    reference_white_Y: float = 1.0,
    return_numpy: bool = False,
    return_sRGB: bool = False
):
    """
    Create a swatch from xyY values (e.g., a row from your CSV).

    - Normalizes Y by reference_white_Y if Y > 1.5 (CSV uses up to ~102.5).
    - Returns PIL Image by default or NumPy array if return_numpy=True.
    """
    srgb = _xyY_to_srgb_clipped(np.array([x, y, Y], dtype=float), reference_white_Y)
    w, h = int(size[0]), int(size[1])
    rgb8 = (np.clip(srgb, 0, 1) * 255.0).round().astype(np.uint8)
    img_arr = np.tile(rgb8.reshape(1, 1, 3), (h, w, 1))
    if return_numpy:
        return img_arr
    if return_sRGB:
        return Image.fromarray(img_arr, mode="RGB"), srgb
    return Image.fromarray(img_arr, mode="RGB")


# Example usage
if __name__ == "__main__":
    # List of Munsell colors to visualize
    munsell_colors = [
        "5R 5/10",  # Red
        "5Y 8/12",  # Yellow
        "5G 5/8",  # Green
        "5B 5/10",  # Blue
        "5P 5/10",  # Purple
        "N 5/",  # Neutral gray
        "10R 4/6",
        "2.5Y 7/8",
        "7.5GY 6/6",
        "10BG 5/4",
    ]

    # plot_munsell_colors(munsell_colors, "Munsell Color Swatches (converted to sRGB)")
    pic = munsell_swatch("10BG 5/4", [224, 224], 1)
    plt.imshow(pic)
    plt.show()
