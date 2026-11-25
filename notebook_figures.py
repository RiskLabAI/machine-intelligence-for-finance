from io import BytesIO

from IPython.display import Image, display
from matplotlib.text import Text
from PIL import Image as RasterImage


NOTEBOOK_FIGURE_WIDTH = 900


def notebook_png(data):
    """Return an image that fits the notebook without stretching the plot."""
    with RasterImage.open(BytesIO(data)) as image:
        if image.width <= NOTEBOOK_FIGURE_WIDTH:
            return data
        height = max(1, round(image.height * NOTEBOOK_FIGURE_WIDTH / image.width))
        preview = image.resize((NOTEBOOK_FIGURE_WIDTH, height), RasterImage.Resampling.LANCZOS)
        buffer = BytesIO()
        preview.save(buffer, format="PNG", compress_level=4)
        return buffer.getvalue()


def figure_preview(figure):
    """Render the figure with screen-sized type while retaining its export style."""
    pixels_per_point = NOTEBOOK_FIGURE_WIDTH / (72 * figure.get_figwidth())
    title_ids = {id(axes.title) for axes in figure.axes}
    title_ids.add(id(getattr(figure, "_suptitle", None)))
    sizes = []
    buffer = BytesIO()
    try:
        for text in figure.findobj(match=Text):
            if not text.get_text():
                continue
            cap = (14 if id(text) in title_ids else 12) / pixels_per_point
            points = text.get_fontsize()
            if points > cap:
                sizes.append((text, points))
                text.set_fontsize(cap)
        figure.savefig(
            buffer,
            format="png",
            dpi=max(120, NOTEBOOK_FIGURE_WIDTH / figure.get_figwidth() * 1.5),
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
            transparent=False,
        )
    finally:
        for text, points in sizes:
            text.set_fontsize(points)
    return notebook_png(buffer.getvalue())


def display_notebook_figure(figure):
    display(Image(data=figure_preview(figure), format="png"))
