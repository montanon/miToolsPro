"""Dynamic plotting constants pulled from Matplotlib."""
from __future__ import annotations

from matplotlib.colors import get_named_colors_mapping
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D


# Helper retrieval functions

def available_colors() -> list[str]:
    """Return the list of named colors known to Matplotlib."""
    return list(get_named_colors_mapping().keys())


def available_markers() -> list[str | int]:
    """Return available marker identifiers."""
    return list(set(MarkerStyle.markers.keys()).union(MarkerStyle.filled_markers))


def available_linestyles() -> list[str | None]:
    """Return available line style strings."""
    return list(Line2D.lineStyles.keys())


def available_colormaps() -> list[str]:
    """Return all registered colormap names."""
    try:
        import matplotlib.pyplot as plt

        return list(plt.colormaps())
    except Exception:  # pragma: no cover - fallback if pyplot not available
        try:
            from matplotlib import colormaps

            return list(colormaps)
        except Exception:  # pragma: no cover - final fallback
            return []


def available_hatches() -> list[str]:
    """Return supported hatch patterns."""
    try:
        from matplotlib.patches import HatchStyle

        return list(HatchStyle.classes.keys())
    except Exception:  # pragma: no cover
        return ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def available_bins() -> list[str]:
    """Return available histogram binning algorithms."""
    return ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]

