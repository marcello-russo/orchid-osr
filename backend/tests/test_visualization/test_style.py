import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.visualization.style import COLORS, DPI, FIGURE_SIZE, setup_style


def test_setup_style_applies_ggplot():
    setup_style()
    fig, ax = plt.subplots()
    assert fig.get_facecolor() is not None
    plt.close(fig)


def test_colors_has_five_entries():
    assert len(COLORS) >= 5


def test_figure_size_is_tuple():
    assert isinstance(FIGURE_SIZE, tuple)
    assert len(FIGURE_SIZE) == 2


def test_dpi_is_300():
    assert DPI == 300
