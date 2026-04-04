"""Interactive matplotlib viewer for explainability."""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from app.explainability.core import inference
from app.explainability.visualization import (
    display_image,
    display_occlusion,
    display_confidence,
    _display_filename,
)


def show_and_test_images(dataset, model, device, class_names, sliding_window_size, stride):
    """Interactive viewer: show images one by one with occlusion heatmaps."""
    plt.ion()
    fig, ax = plt.subplots(1, len(class_names) + 1, figsize=(18, 3))
    plt.subplots_adjust(bottom=0.2)
    model.to(device)

    class InteractiveViewer:
        def __init__(self):
            self.dataset_iter = iter(dataset)
            self.current_image = None
            self.current_label = None
            self.current_filename = None
            self.confidence_text = fig.text(0.01, 0.1, '', ha='left', va='center', fontsize=8)
            self.occlusion_text_x, self.occlusion_text_y = ax[1].get_position().x0, ax[1].get_position().y0 - 0.1
            self.occlusion_text = fig.text(self.occlusion_text_x, self.occlusion_text_y, '', ha='left', va='center', fontsize=9)

        def display_next(self, event=None):
            try:
                data = next(self.dataset_iter)
                if isinstance(data, tuple) and len(data) == 3:
                    image, label, filename = data
                elif isinstance(data, tuple) and len(data) == 2:
                    image, filename = data
                    label = None
                else:
                    image, label, filename = data, None, None

                self.current_image = image.to(device)
                self.current_label = label.item() if label is not None else None
                self.current_filename = filename

                prediction_values, predicted = inference(model, self.current_image.unsqueeze(0), device)

                display_image(ax[0], self.current_image, predicted, self.current_label, class_names)
                for idx, _ in enumerate(class_names):
                    display_occlusion(fig, ax[idx + 1], self.current_image, model, idx, self.occlusion_text, sliding_window_size, stride, class_names)
                display_confidence(self.confidence_text, prediction_values, class_names)
                _display_filename(fig.text(0.95, 0.02, '', ha='right', va='center', fontsize=9), self.current_filename)

                bg_color = 'white' if self.current_label is not None and predicted == self.current_label else 'lightcoral'
                fig.patch.set_facecolor(bg_color)

                fig.canvas.draw_idle()
            except StopIteration:
                print("Fine del dataset.")
                plt.close(fig)

    viewer = InteractiveViewer()
    viewer.display_next()

    ax_button = plt.axes([0.85, 0.05, 0.1, 0.075])
    next_button = Button(ax_button, 'Next')
    next_button.on_clicked(viewer.display_next)

    plt.show(block=True)
