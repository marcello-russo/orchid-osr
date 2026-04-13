"""Visualization functions for model explainability."""
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from captum.attr import IntegratedGradients, Occlusion
from captum.attr._utils.visualization import _normalize_attr

import app.preprocessing.normalization as normalization
from app.explainability.core import inference, get_percentages


def display_image(ax, image, predicted, label, class_names):
    """Display the original image with predicted and true labels."""
    image = normalization.denormalize_image(
        image, normalization.get_mean(), normalization.get_std()
    ).cpu()
    ax.clear()
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis('off')
    predicted_text = f'Predicted: {class_names[predicted.item()]}'
    label_text = f'Label: {class_names[label]}' if label is not None else 'Label: Undefined'
    ax.set_title(f'{predicted_text}\n{label_text}', fontsize=12)


def display_confidence(text, values, class_names):
    """Display class confidence percentages in a text element."""
    confidence_text = 'Attributions:\n'
    percentages = get_percentages(values)
    for i, percentage in enumerate(percentages.squeeze()):
        confidence_text += f'{class_names[i]}: {percentage:.2f}%\n'
    text.set_text(confidence_text)


def _display_filename(text, filename):
    """Display filename in a text element."""
    text.set_text(f'Filename: {filename}' if filename else '')


def display_integrated_gradients(ax, image, ig, label, class_names):
    """Display integrated gradients attribution heatmap."""
    attributions = ig.attribute(image.unsqueeze(0), target=label)
    attribution_map = np.sum(np.abs(attributions.squeeze(0).cpu().numpy()), axis=0)
    ax.clear()
    im = ax.imshow(attribution_map, cmap='hot')
    ax.set_title(f'IntegratedGradients attributions\nfor class {class_names[label]}')
    ax.axis('off')
    if not hasattr(ax, 'colorbar') or ax.colorbar is None:
        cbar = plt.colorbar(im, ax=ax)
        ax.colorbar = cbar


def get_attribution_class_map(global_attribution):
    """Compute a per-pixel class map from occlusion attributions."""
    global_attribution = global_attribution.squeeze(0).cpu().detach().numpy()
    class_map = np.argmax(global_attribution, axis=0)
    return class_map


def _add_occlusion_hover_text(fig, ax, class_map, global_attribution, occlusion_text, class_names):
    """Attach a hover handler that shows per-class occlusion scores."""
    def update_occlusion_text(x, y):
        if 0 <= x < class_map.shape[2] and 0 <= y < class_map.shape[1]:
            class_id = int(class_map[0, y, x])
            actual_class_vector = global_attribution[:, 0, y, x].cpu().detach().numpy()
            ordered_classes = np.argsort(actual_class_vector)[::-1]
            occlusion_text_str = ''
            for i in range(len(class_names)):
                occlusion_text_str += f'{class_names[ordered_classes[i]]}: {actual_class_vector[ordered_classes[i]]:.2f}\n'
            occlusion_text.set_text(occlusion_text_str)
        else:
            occlusion_text.set_text('')

    def on_hover(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            update_occlusion_text(x, y)

    fig.canvas.mpl_connect('motion_notify_event', on_hover)


def display_occlusion(fig, ax, image, model, label, occlusion_text, sliding_window_size, stride, class_names):
    """Display occlusion attribution heatmap with interactive hover."""
    occlusion = Occlusion(model)

    global_attribution = occlusion.attribute(
        image.unsqueeze(0),
        sliding_window_shapes=(3, sliding_window_size, sliding_window_size),
        strides=(3, stride, stride)
    )

    class_map = get_attribution_class_map(global_attribution)
    label_attributions = global_attribution[label, :, :, :]

    label_attributions = label_attributions.squeeze(0).cpu().detach().numpy()
    label_attributions = np.transpose(label_attributions, (1, 2, 0))

    image_np = normalization.denormalize_image(
        image, normalization.get_mean(), normalization.get_std()
    ).cpu().numpy().transpose(1, 2, 0)

    norm_attr = _normalize_attr(label_attributions, sign="all", outlier_perc=2, reduction_axis=2)
    cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    vmin, vmax = -1, 1

    ax.clear()
    ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    ax.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)
    ax.set_title('Occlusion attributions\nfor class ' + class_names[label], fontsize=8)
    ax.axis('off')

    _add_occlusion_hover_text(fig, ax, class_map, global_attribution, occlusion_text, class_names)


def display_occlusion_for_save(ax, image, model, label, sliding_window_size, stride, class_names):
    """Display occlusion attribution heatmap for saving (no hover interaction)."""
    occlusion = Occlusion(model)

    attributions = occlusion.attribute(
        image.unsqueeze(0),
        target=label,
        sliding_window_shapes=(3, sliding_window_size, sliding_window_size),
        strides=(3, stride, stride)
    )

    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))

    image_np = normalization.denormalize_image(
        image, normalization.get_mean(), normalization.get_std()
    ).cpu().numpy().transpose(1, 2, 0)

    norm_attr = _normalize_attr(attributions, sign="all", outlier_perc=2, reduction_axis=2)
    cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    vmin, vmax = -1, 1

    ax.clear()
    im = ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    im_attr = ax.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)
    ax.set_title(f'Occlusion attributions\nfor class {class_names[label]}')
    ax.axis('off')

    if not hasattr(ax, 'colorbar') or ax.colorbar is None:
        cbar = plt.colorbar(im_attr, ax=ax)
        ax.colorbar = cbar


def generate_output_images(dataset, model, device, class_names, output_dir, sliding_window_size, stride):
    """Generate and save explainability images for every sample in the dataset."""
    model.to(device)
    ig = IntegratedGradients(model)
    os.makedirs(output_dir, exist_ok=True)
    class_counters = defaultdict(int)

    # Map dataset class indices to model class indices
    model_class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    dataset_to_model = {}
    for ds_idx, cls_name in enumerate(dataset.classes):
        if cls_name in model_class_to_idx:
            dataset_to_model[ds_idx] = model_class_to_idx[cls_name]

    for idx, data in enumerate(dataset):
        if isinstance(data, tuple):
            if len(data) == 3:
                image, label, filename = data
            elif len(data) == 2:
                image, filename = data
                label = None
            else:
                image, label, filename = data, None, None
        else:
            image, label, filename = data, None, None

        if isinstance(label, torch.Tensor):
            label = label.item()

        image = image.to(device)

        # Map dataset label to model class index
        model_label = dataset_to_model.get(label) if label is not None else None

        label_dir = os.path.join(output_dir, f"{dataset.classes[label]}" if label is not None else "unlabeled")
        os.makedirs(label_dir, exist_ok=True)

        if label is not None:
            class_counters[label] += 1
            class_index = class_counters[label]
        else:
            class_counters["unlabeled"] += 1
            class_index = class_counters["unlabeled"]

        prediction_values, predicted = inference(model, image.unsqueeze(0), device)

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)
        display_image(ax[0], image, predicted, model_label, class_names)
        display_occlusion_for_save(ax[1], image, model, predicted if model_label is None else model_label, sliding_window_size, stride, class_names)
        display_integrated_gradients(ax[2], image, ig, predicted if model_label is None else model_label, class_names)
        display_confidence(fig.text(0.01, 0.1, '', ha='left', va='center', fontsize=9), prediction_values, class_names)
        _display_filename(fig.text(0.95, 0.02, '', ha='right', va='center', fontsize=9), filename)

        bg_color = 'white' if model_label is not None and predicted == model_label else 'lightcoral'
        fig.patch.set_facecolor(bg_color)

        output_path = os.path.join(label_dir, f"{class_index}_{filename[:-4]}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Immagine salvata: {output_path}")

    print("Tutte le immagini sono state elaborate e salvate.")
