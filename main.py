from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_hex
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    SamModel,
    SamProcessor,
)
import pandas as pd
from skimage import color
import colorsys
import os
import glob
from sklearn.cluster import KMeans
import functools


def handle_processing_errors(operation_name, fallback_value=None):
    """Decorator for consistent error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {operation_name}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator


class FashionColorExtractor:
    def __init__(self):
        self.device = self._initialize_device()
        self.detector_processor, self.detector = self._load_detection_models()
        self.sam_model, self.sam_processor = self._load_sam_models()
        self.object_counter = 0

    def _initialize_device(self):
        """Initialize the appropriate computing device"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def _load_detection_models(self):
        """Load object detection models"""
        detector_ckpt = "yainage90/fashion-object-detection"
        processor = AutoImageProcessor.from_pretrained(detector_ckpt)
        model = AutoModelForObjectDetection.from_pretrained(detector_ckpt).to(
            self.device
        )
        return processor, model

    def _load_sam_models(self):
        """Load SAM segmentation models"""
        sam_ckpt = "facebook/sam-vit-base"
        model = SamModel.from_pretrained(sam_ckpt).to(self.device)
        processor = SamProcessor.from_pretrained(sam_ckpt)
        return model, processor

    def _convert_lab_color(self, lab_color, output_format="rgb"):
        """Unified color conversion with error handling"""
        try:
            lab_array = np.array(lab_color).reshape(1, 1, 3)
            rgb_array = color.lab2rgb(lab_array)
            rgb_normalized = np.clip(rgb_array.flatten(), 0, 1)
            
            if output_format == "rgb":
                return rgb_normalized
            elif output_format == "hex":
                rgb_255 = (rgb_normalized * 255).astype(int)
                return "#{:02x}{:02x}{:02x}".format(*rgb_255)
        except Exception as e:
            print(f"Warning: LAB conversion failed for {lab_color}: {e}")
            return np.array([0.5, 0.5, 0.5]) if output_format == "rgb" else "#808080"

    def lab_to_rgb(self, lab_color):
        """Convert CIELAB color to RGB (0-1 range)"""
        return self._convert_lab_color(lab_color, "rgb")

    def lab_to_hex(self, lab_color):
        """Convert CIELAB color to hexadecimal format"""
        return self._convert_lab_color(lab_color, "hex")

    def _prepare_inputs_for_device(self, inputs):
        """Handle device-specific tensor preparation"""
        if self.device.type == "mps":
            return {
                k: (
                    v.to(self.device, dtype=torch.float32)
                    if v.dtype.is_floating_point
                    else v.to(self.device)
                )
                for k, v in inputs.items()
            }
        else:
            return inputs.to(self.device)

    def _run_model_inference(self, model, inputs):
        """Handle device-specific model inference"""
        if self.device.type == "mps":
            with torch.no_grad():
                return model(**inputs)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return model(**inputs)

    def _get_visualization_color(self, index, color_type="detection"):
        """Get consistent colors for visualization elements"""
        color_palettes = {
            "detection": [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "cyan",
                "yellow",
                "pink",
            ],
        }
        if color_type == "detection":
            colors = color_palettes["detection"]
            return colors[index % len(colors)]
        else:
            return plt.cm.Set1(index / 8)  # Normalize for matplotlib

    def _find_image_files(self, directory_path):
        """Find all image files in directory"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []

        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, extension)))
            image_files.extend(
                glob.glob(os.path.join(directory_path, extension.upper()))
            )
        return image_files

    def _extract_color_attributes(self, color_info):
        """Extract and format all color space attributes"""
        cielab_values = color_info["cielab"]
        if "rgb" in color_info:
            rgb_values = color_info["rgb"]
        else:
            rgb_values = self.lab_to_rgb(cielab_values)
        if "hsv" in color_info:
            hsv_values = color_info["hsv"]
        else:
            hsv_raw = colorsys.rgb_to_hsv(*rgb_values)
            hsv_values = [hsv_raw[0] * 360, hsv_raw[1], hsv_raw[2]]
        hex_color = color_info.get("hex", self.lab_to_hex(cielab_values))
        return {
            "cielab": cielab_values,
            "rgb": rgb_values,
            "hsv": hsv_values,
            "hex": hex_color,
        }

    def _process_single_color(self, lab_color, rgb_color, frequency):
        """Process a single color through the complete pipeline"""
        hsv_color = colorsys.rgb_to_hsv(*rgb_color)
        hsv_degrees = [hsv_color[0] * 360, hsv_color[1], hsv_color[2]]
        l, a, b = lab_color
        chroma = np.sqrt(a**2 + b**2)
        return {
            "cielab": tuple(lab_color.round(2)),
            "rgb": tuple(rgb_color.round(3)),
            "hsv": tuple(
                [
                    round(hsv_degrees[0], 1),
                    round(hsv_degrees[1], 3),
                    round(hsv_degrees[2], 3),
                ]
            ),
            "hex": self.lab_to_hex(lab_color),
            "chroma": round(chroma, 2),
            "percentage": frequency * 100,
        }

    def _create_color_row_data(
        self, object_info, color_info, collection_id, image_filename
    ):
        """Create a single row of color data for the DataFrame"""
        color_attrs = self._extract_color_attributes(color_info)
        row_data = {
            "object_id": object_info["object_id"],
            "collection_id": collection_id,
            "object_name": object_info["object_name"],
            "cielab_l": round(color_attrs["cielab"][0], 2),
            "cielab_a": round(color_attrs["cielab"][1], 2),
            "cielab_b": round(color_attrs["cielab"][2], 2),
            "rgb_r": round(color_attrs["rgb"][0], 3),
            "rgb_g": round(color_attrs["rgb"][1], 3),
            "rgb_b": round(color_attrs["rgb"][2], 3),
            "hsv_h": round(color_attrs["hsv"][0], 1),
            "hsv_s": round(color_attrs["hsv"][1], 3),
            "hsv_v": round(color_attrs["hsv"][2], 3),
            "hex_color": color_attrs["hex"],
            "chroma": color_info.get("chroma"),
            "percentage": round(color_info["percentage"], 2),
        }
        if image_filename:
            row_data["image_filename"] = image_filename
        if "hue_family" in color_info:
            row_data["hue_family"] = color_info["hue_family"]
            row_data["num_original_colors"] = color_info.get("num_original_colors", 1)
        return row_data

    @handle_processing_errors("fashion item detection", [])
    def detect_fashion_items(self, image):
        """Detect fashion items and return bounding boxes"""
        with torch.no_grad():
            inputs = self.detector_processor(images=[image], return_tensors="pt")
            outputs = self.detector(**inputs.to(self.device))
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = self.detector_processor.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=target_sizes
            )[0]
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append(
                {
                    "score": score.item(),
                    "label": self.detector.config.id2label[label.item()],
                    "box": [coord.item() for coord in box],  # [x1, y1, x2, y2]
                }
            )
        return detections

    @handle_processing_errors("SAM segmentation", np.array([]))
    def segment_with_sam(self, image, bbox):
        """Use SAM to get precise segmentation mask from bounding box"""
        input_boxes = [[[bbox[0], bbox[1], bbox[2], bbox[3]]]]
        inputs = self.sam_processor(image, input_boxes=input_boxes, return_tensors="pt")
        prepared_inputs = self._prepare_inputs_for_device(inputs)
        outputs = self._run_model_inference(self.sam_model, prepared_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][0][0].numpy()  # Shape: (H, W)

    def assign_hue_family_hsv(self, rgb_color):
        """Hue family assignment using HSV with beige detection"""
        hsv = colorsys.rgb_to_hsv(*rgb_color)
        h, s, v = hsv[0] * 360, hsv[1], hsv[2]
        # Very low saturation or very dark = neutral (but with pastel exception)
        if (s < 0.12 or v < 0.1) and not (v > 0.85 and s >= 0.08):
            return "neutral"
        # Dark colors with medium saturation = neutral (charcoal, dark grays)
        if v < 0.35 and s < 0.5:
            return "neutral"
        # Special case: Sage greens (desaturated greens that should stay green)
        if 100 <= h <= 140 and 0.08 <= s <= 0.2 and v >= 0.6:
            return "green"
        # Special case: Pastels (very bright with low-medium saturation)
        if v > 0.85 and 0.08 <= s <= 0.25:
            if h < 20 or h >= 330:
                return "magenta"
            elif h < 75:
                return "yellow"
            elif h < 165:
                return "green"
            elif h < 255:
                return "blue"
            elif h < 330:
                return "purple"
        # Special case: Dark reds (burgundy, wine, etc.)
        if (h < 20 or h >= 340) and v <= 0.6 and s >= 0.12:
            return "red"
        # Special case: Olive detection
        if (50 <= h <= 70) and (s >= 0.8) and (v <= 0.6):
            return "green"
        # Beige detection
        if (10 <= h <= 60) and (0.12 <= s <= 0.45) and (0.4 <= v <= 0.85):
            return "beige"
        # Brown detection
        if 15 <= h < 60 and s >= 0.12 and (v < 0.4 or (s >= 0.35 and v < 0.75)):
            return "brown"
        # Standard hue ranges (for more saturated colors)
        if s >= 0.25:
            if h < 15 or h >= 340:
                return "red"
            elif h < 45:
                return "orange"
            elif h < 75:
                return "yellow"
            elif h < 165:
                return "green"
            elif h < 255:
                return "blue"
            elif h < 285:
                return "purple"
            elif h < 340:
                return "magenta"
        # For low-medium saturation colors that aren't beige or brown
        else:
            if (h < 30 or h >= 330) and s >= 0.12:
                return "magenta"
            elif 30 <= h < 90 and s >= 0.15:
                return "beige"
            elif 90 <= h < 150 and s >= 0.12:
                return "green"
            elif 150 <= h < 270 and s >= 0.15:
                return "blue"
            elif 270 <= h < 330 and s >= 0.15:
                return "purple"
            else:
                return "neutral"

    def is_neutral(self, l, a, b, chroma_threshold=10):
        """Determine if a color is neutral based on chroma"""
        chroma = np.sqrt(a**2 + b**2)
        return chroma < chroma_threshold

    def aggregate_colors_by_hue_family(self, colors, min_percentage_threshold=3.0):
        """Aggregate colors within the same hue family by weighted average"""
        processed_colors = []
        for color in colors:
            l, a, b = color["cielab"]
            rgb = color["rgb"]
            hsv = color["hsv"]
            percentage = color["percentage"]
            hue_family = self.assign_hue_family_hsv(rgb)
            chroma = np.sqrt(a**2 + b**2)
            processed_colors.append(
                {
                    "l": l,
                    "a": a,
                    "b": b,
                    "r": rgb[0],
                    "g": rgb[1],
                    "b_rgb": rgb[2],
                    "h": hsv[0],
                    "s": hsv[1],
                    "v": hsv[2],
                    "percentage": percentage,
                    "hue_family": hue_family,
                    "chroma": chroma,
                }
            )
        df = pd.DataFrame(processed_colors)
        aggregated = []
        for family in df["hue_family"].unique():
            family_colors = df[df["hue_family"] == family]
            total_percentage = family_colors["percentage"].sum()
            if total_percentage >= min_percentage_threshold:
                weights = family_colors["percentage"].values
                # Calculate weighted averages
                weighted_averages = {
                    col: np.average(family_colors[col], weights=weights)
                    for col in ["l", "a", "b_rgb", "r", "g", "h", "s", "v"]
                }
                avg_chroma = np.sqrt(
                    weighted_averages["a"] ** 2 + weighted_averages["b_rgb"] ** 2
                )
                aggregated.append(
                    {
                        "hue_family": family,
                        "cielab": [
                            weighted_averages["l"],
                            weighted_averages["a"],
                            weighted_averages["b_rgb"],
                        ],
                        "rgb": [
                            weighted_averages["r"],
                            weighted_averages["g"],
                            weighted_averages["b_rgb"],
                        ],
                        "hsv": [
                            weighted_averages["h"],
                            weighted_averages["s"],
                            weighted_averages["v"],
                        ],
                        "chroma": avg_chroma,
                        "percentage": total_percentage,
                        "num_original_colors": len(family_colors),
                    }
                )
        return sorted(aggregated, key=lambda x: x["percentage"], reverse=True)

    @handle_processing_errors("color extraction", [])
    def extract_colors_from_mask(
        self, image, mask, k=10, aggregate_by_hue=True, min_percentage_threshold=3.0
    ):
        """Extract dominant colors from segmented region using CIELAB color space"""
        image_np = np.array(image)
        segmented_pixels = image_np[mask]
        if len(segmented_pixels) == 0:
            return []
        pixels_rgb = segmented_pixels.reshape(-1, 3) / 255.0
        valid_pixels = pixels_rgb[~np.any(pixels_rgb < 0, axis=1)]
        if len(valid_pixels) < k:
            k = len(valid_pixels)
        if k == 0:
            return []
        # Convert RGB to CIELAB for clustering
        rgb_for_conversion = valid_pixels.reshape(1, -1, 3)
        lab_pixels = color.rgb2lab(rgb_for_conversion).reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(lab_pixels)
        lab_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        color_frequencies = []
        for i, lab_color in enumerate(lab_centers):
            frequency = np.sum(labels == i) / len(labels)
            rgb_color = self.lab_to_rgb(lab_color)
            color_data = self._process_single_color(lab_color, rgb_color, frequency)
            color_frequencies.append(color_data)
        color_frequencies.sort(key=lambda x: x["percentage"], reverse=True)
        if aggregate_by_hue:
            color_frequencies = self.aggregate_colors_by_hue_family(
                color_frequencies, min_percentage_threshold
            )
        return color_frequencies

    def _create_color_swatch(
        self, fig, color_info, x_pos, y_pos, swatch_size, has_hue_families
    ):
        """Create a single color swatch with label"""
        # Determine hex color
        if "rgb" in color_info:
            rgb = color_info["rgb"]
            rgb_255 = [int(np.clip(c * 255, 0, 255)) for c in rgb]
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_255)
        elif "hex" in color_info:
            hex_color = color_info["hex"]
        elif "cielab" in color_info:
            hex_color = self.lab_to_hex(color_info["cielab"])
        else:
            hex_color = "#808080"
        percentage = color_info.get("percentage", 0)
        # Create swatch
        swatch = patches.Rectangle(
            (x_pos, y_pos),
            swatch_size,
            swatch_size,
            facecolor=hex_color,
            edgecolor="black",
            linewidth=1,
            transform=fig.transFigure,
        )
        fig.patches.append(swatch)
        # Create label
        if has_hue_families and "hue_family" in color_info:
            hue_family = color_info["hue_family"]
            num_original = color_info.get("num_original_colors", 1)
            if num_original > 1:
                label = f"{hue_family}\n{percentage:.1f}%\n({num_original} colors)"
            else:
                label = f"{hue_family}\n{percentage:.1f}%"
        else:
            if "hsv" in color_info:
                hsv = color_info["hsv"]
                label = f"H:{hsv[0]:.0f}°\nS:{hsv[1]:.2f}\n{percentage:.1f}%"
            else:
                label = f"{percentage:.1f}%"
        fig.text(
            x_pos + swatch_size / 2,
            y_pos - 0.01,
            label,
            fontsize=9,
            ha="center",
            va="top",
            transform=fig.transFigure,
        )

    def visualize_results(self, image_path, results, save_path=None):
        """Create visualization with bounding boxes, labels, masks, and color palettes"""
        image = Image.open(image_path).convert("RGB")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title("Object Detection Results", fontsize=16, fontweight="bold")
        ax1.axis("off")
        ax2.imshow(image)
        ax2.set_title("Segmentation Masks", fontsize=16, fontweight="bold")
        ax2.axis("off")
        # Add detection boxes and segmentation masks
        for i, result in enumerate(results):
            bbox = result["bbox"]
            item = result["object_name"]
            confidence = result["confidence"]
            detection_color = self._get_visualization_color(i, "detection")
            # Detection rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=3,
                edgecolor=detection_color,
                facecolor="none",
            )
            ax1.add_patch(rect)
            # Label
            label_text = f"{item}\n{confidence:.2f}"
            ax1.text(
                bbox[0],
                bbox[1] - 10,
                label_text,
                fontsize=12,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor=detection_color, alpha=0.8
                ),
            )
            # Segmentation mask
            if "mask" in result:
                mask = result["mask"]
                colored_mask = np.zeros((*mask.shape, 4))
                color_rgba = plt.cm.Set1(i / len(results))
                colored_mask[mask] = color_rgba
                ax2.imshow(colored_mask, alpha=0.6)
        # Add color palettes
        self._add_color_palettes(fig, results)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")
        plt.show()
        return fig

    def _add_color_palettes(self, fig, results):
        """Add color palettes to the visualization figure"""
        palette_y_start = 0.02
        swatch_size = 0.03
        swatch_spacing = 0.05
        start_x = 0.02
        for i, result in enumerate(results):
            colors_info = result["colors"][:3]
            if not colors_info:
                continue
            current_y = palette_y_start + i * 0.15
            has_hue_families = colors_info and "hue_family" in colors_info[0]
            # Add title
            title_suffix = (
                " (HSV-based hue families)"
                if has_hue_families
                else " (individual colors)"
            )
            fig.text(
                start_x,
                current_y + 0.08,
                f"{result['object_name']}{title_suffix}:",
                fontsize=12,
                fontweight="bold",
                transform=fig.transFigure,
            )
            # Add color swatches
            for j, color_info in enumerate(colors_info):
                swatch_x = start_x + j * swatch_spacing
                self._create_color_swatch(
                    fig, color_info, swatch_x, current_y, swatch_size, has_hue_families
                )

    def create_dataframe(self, results, collection_id=1, image_filename=None):
        """Create a pandas DataFrame from the results in long format"""
        data_rows = []
        for result in results:
            object_info = {
                "object_id": result["object_id"],
                "object_name": result["object_name"],
            }
            for color_info in result["colors"]:
                row_data = self._create_color_row_data(
                    object_info, color_info, collection_id, image_filename
                )
                data_rows.append(row_data)
        return pd.DataFrame(data_rows)

    @handle_processing_errors("image processing", ([], pd.DataFrame()))
    def process_image(
        self,
        image_path,
        collection_id=1,
        visualize=True,
        save_path=None,
        csv_output_path=None,
        k_colors=10,
        aggregate_by_hue=True,
        min_percentage_threshold=3.0,
    ):
        """Complete pipeline: detect -> segment -> extract colors -> create DataFrame"""
        image = Image.open(image_path).convert("RGB")
        detections = self.detect_fashion_items(image)
        results = []
        for detection in detections:
            self.object_counter += 1
            object_id = self.object_counter
            mask = self.segment_with_sam(image, detection["box"])
            colors = self.extract_colors_from_mask(
                image,
                mask,
                k=k_colors,
                aggregate_by_hue=aggregate_by_hue,
                min_percentage_threshold=min_percentage_threshold,
            )
            results.append(
                {
                    "object_id": object_id,
                    "object_name": detection["label"],
                    "confidence": detection["score"],
                    "bbox": detection["box"],
                    "mask": mask,
                    "colors": colors,
                }
            )
        image_filename = os.path.basename(image_path)
        if results:
            df = self.create_dataframe(results, collection_id, image_filename)
            if visualize:
                self.visualize_results(image_path, results, save_path)
            return results, df
        else:
            print(f"No fashion items detected in {image_filename}.")
            return [], pd.DataFrame()

    def process_directory(
        self,
        directory_path,
        collection_id=1,
        visualize=True,
        csv_output_path=None,
        k_colors=10,
        aggregate_by_hue=True,
        min_percentage_threshold=3.0,
    ):
        """Process all images in a directory and return combined results"""
        image_files = self._find_image_files(directory_path)
        if not image_files:
            print(f"No image files found in directory: {directory_path}")
            return [], pd.DataFrame()
        print(f"Found {len(image_files)} image files in directory: {directory_path}")
        # Select random image for visualization
        visualization_image = None
        if visualize and image_files:
            import random

            visualization_image = random.choice(image_files)
            print(
                f"Selected random image for visualization: {os.path.basename(visualization_image)}"
            )
        all_results = []
        all_dataframes = []
        visualization_results = None
        for i, image_path in enumerate(image_files, 1):
            print(
                f"\nProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}"
            )
            is_visualization_image = (
                visualization_image and image_path == visualization_image
            )
            results, df = self.process_image(
                image_path,
                collection_id=collection_id,
                visualize=False,
                save_path=None,
                csv_output_path=None,
                k_colors=k_colors,
                aggregate_by_hue=aggregate_by_hue,
                min_percentage_threshold=min_percentage_threshold,
            )
            if results:
                all_results.extend(results)
                all_dataframes.append(df)
                print(f"  - Detected {len(results)} objects")
                if is_visualization_image:
                    visualization_results = results
            else:
                print(f"  - No fashion items detected")
        return self._finalize_directory_processing(
            all_dataframes,
            all_results,
            image_files,
            csv_output_path,
            visualize,
            visualization_image,
            visualization_results,
        )

    def _finalize_directory_processing(
        self,
        all_dataframes,
        all_results,
        image_files,
        csv_output_path,
        visualize,
        visualization_image,
        visualization_results,
    ):
        """Finalize directory processing with summary and visualization"""
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"\nProcessing complete!")
            print(f"Total images processed: {len(image_files)}")
            print(f"Total objects detected: {len(all_results)}")
            print(f"Total color entries in dataframe: {len(combined_df)}")
            if csv_output_path:
                combined_df.to_csv(csv_output_path, index=False)
                print(f"Combined CSV saved to: {csv_output_path}")
            if visualize and visualization_image and visualization_results:
                print(
                    f"\nCreating visualization for randomly selected image: {os.path.basename(visualization_image)}"
                )
                self.visualize_results(
                    visualization_image,
                    visualization_results,
                    save_path="fashion_analysis_sample.png",
                )
            return all_results, combined_df
        else:
            print("No data to combine - no fashion items detected in any images.")
            return [], pd.DataFrame()


# Usage example
if __name__ == "__main__":
    import os

    # Specify directory containing images
    input_directory = "images"  # Directory in the root of the project
    if not os.path.exists(input_directory):
        print(f"Warning: Input directory '{input_directory}' not found.")
        print(
            "Please create a directory named 'images' in the current directory and add your image files to it."
        )
        print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        exit(1)
    print("Initializing Fashion Color Extractor...")
    extractor = FashionColorExtractor()
    print(f"Processing all images in directory: {input_directory}")
    results, df = extractor.process_directory(
        input_directory,
        collection_id=1,
        visualize=True,
        csv_output_path="fashion_colors_combined.csv",
        k_colors=10,
        aggregate_by_hue=True,
        min_percentage_threshold=3.0,
    )
    if not df.empty:
        print("\n" + "=" * 50)
        print("COMBINED DATAFRAME SUMMARY")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nObjects detected: {df['object_id'].nunique()}")
        print(
            f"Images processed: {df['image_filename'].nunique() if 'image_filename' in df.columns else 'N/A'}"
        )
        print(f"Total color entries: {len(df)}")
        print("\nObject breakdown by image:")
        if "image_filename" in df.columns:
            for image_name in df["image_filename"].unique():
                image_data = df[df["image_filename"] == image_name]
                objects_in_image = image_data["object_id"].nunique()
                colors_in_image = len(image_data)
                print(
                    f"  {image_name}: {objects_in_image} objects, {colors_in_image} color entries"
                )
        print("\nSample of combined data:")
        print(df.head(15).to_string(index=False))
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    if results:
        object_types = {}
        for result in results:
            obj_type = result["object_name"]
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        print(f"Total objects detected across all images: {len(results)}")
        print("\nObject type breakdown:")
        for obj_type, count in sorted(object_types.items()):
            print(f"  {obj_type}: {count}")
    else:
        print("No fashion items detected in any images.")
