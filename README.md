# color-to-table

Extract structured color data from fashion images using object detection and segmentation ML models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Results](#example-results)
- [Detailed Usage](#detailed-usage)
- [Limitations & Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Overview

Analyzing color features in fashion runway photos can reveal trends that can drive decisions in design, marketing, and retail. This part of the analysis process is often time-consuming and error-prone, making it well-suited for automation.\
\
Using a pair of computer vision models, this script takes a directory of fashion runway images as input, and outputs a CSV containing structured data on colors. The output table uses long format for easier analysis.

## Features

- Custom color taxonomy and hue classification heuristics.
- Structured output data including CIELAB, RGB, HSV, and hexadecimal values.
- Uses domain-specific model for object detection.

## Use cases

- Trend analysis across many levels of granularity
- Automated tagging of product colors for e-commerce
- Color-based recommendation systems

## Requirements

- Python >= 3.13
- ~1 GB storage

This project was originally assembled on an M2 Macbook Air with 8 GB of RAM. As such, it currently uses optimizations specific to Apple Silicon. That said, any laptop or desktop computer with similar (or even somewhat lesser) specs should be able to handle the inference easily.\
\
Use of a GPU is not required.

## Installation

- Clone the repository:

```bash
git clone [url]
cd color-to-table
```

This repository downloads with files ready for `pip` or `uv` to use. If using `pip`, create the virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If using `uv`, simply sync while in the project's root directory:

```bash
uv sync
```

## Quick Start

- If using pip, be sure to activate the virtual environment.
- Images for running inference should be placed in the `images` directory. The cloned repository comes with an example image.
- After inference images are in the `images` directory, run the following:

For `pip`:

```bash
python3 main.py
```

For `uv`:

```bash
uv run main.py
```

## Example Results

### Input

[inline image]

### Outputs

[inline image]

|object_id|collection_id|object_name|cielab_l|cielab_a|cielab_b|rgb_r|rgb_g|rgb_b|hsv_h|hsv_s|hsv_v|hex_color|chroma            |percentage|image_filename          |hue_family|num_original_colors|
|---------|-------------|-----------|--------|--------|--------|-----|-----|-----|-----|-----|-----|---------|------------------|----------|------------------------|----------|-------------------|
|2        |1            |bottom     |83.95   |-2.33   |0.83    |0.799|0.828|0.832|184.2|0.042|0.832|#cdd2cf  |2.4711734772041427|87.89     |fashion-show-1746592.jpg|neutral   |6                  |
|2        |1            |bottom     |86.5    |3.99    |0.66    |0.948|0.833|0.664|35.7 |0.3  |0.948|#e0d5d7  |4.044872803933395 |4.17      |fashion-show-1746592.jpg|orange    |1                  |
|2        |1            |bottom     |73.41   |7.46    |0.5     |0.832|0.682|0.495|33.3 |0.405|0.832|#c2afb3  |7.476404550316951 |4.03      |fashion-show-1746592.jpg|beige     |1                  |
|2        |1            |bottom     |53.12   |10.93   |0.31    |0.627|0.468|0.311|29.2 |0.504|0.627|#91787e  |10.933346587406001|3.9       |fashion-show-1746592.jpg|brown     |2                  |

*(Table edited to results about shorts only for length)

## Detailed Usage

### File Formats

#### Supported Image Inputs

- JPEG
- PNG
- BMP
- TIFF

#### Output Formats

- CSV (structured data)
- PNG (sample visualization)

### Configuration Options

- `input_directory`: Specifies custom path to directory for input images
- `collection_id`: Sets a unique collection identifier for the batch to be processed
- `visualize`: Chooses one input image at random to show object detection results (bounding box, object name, confidence score), segmentation bounds, and color extractions. Outputs visualization in both a post-extraction popup window a PNG in the root directory.
- `csv_output_path`: Customizes the name of the CSV to be produced in the root directory. Be sure this string includes the .csv extension.
- `k_colors`: Maximum number of colors to be detected per object.
- `aggregate_by_hue`: Group colors according to custom hue family taxonomy
- `min_percentage_threshold`: Threshold of coverage percentage to warrant a color's inclusion in extraction and/or aggregation.

### Data Schema

#### Core Columns

- `object_id` (int): Unique identifier for each detected object
- `collection_id` (int): Identifier for the image collection/batch
- `object_name` (str): Type of fashion item detected
- `cielab_l` (float): CIELAB Lightness (0 to 100)
- `cielab_a` (float): CIELAB Green-Red axis (-128 to 127)
- `cielab_b` (float): CIELAB Blue-Yellow axis (-128 to 127)
- `rgb_r` (float): RGB red component (0 to 1)
- `rgb_g` (float): RGB green component (0 to 1)
- `rgb_b` (float): RGB blue component (0 to 1)
- `hsv_h` (float): Hue in degrees (0 to 360)
- `hsv_s` (float): Saturation (0 to 1)
- `hsv_v` (float): Value/Brightness (0 to 1)
- `hex_color` (str): Hexadecimal color code
- `chroma` (float): Color intensity/saturation in CIELAB
- `percentage` (float): Percentage of this color in the object (as a number from 0 to 100, not a proportion from 0 to 1)

#### Conditional Columns

- `image_filename` (str): Name of source image file
- `hue_family` (str): Human-readable color family classification
- `num_original_colors` (int): Number of original k-means colors aggregated

## Limitations and Future Work

- Object detection accuracy is currently limited, leading to error propagation. Currently testing alternative approaches to reduce this.
- Output could be extended to SQL databases for more robust storage practices and analysis techniques.
- Script could be broken into more parts and configuration options could be turned into flags, all for better modularity.
- Hue family naming and color aggregation steps could be split into their own optional scripts, or made part of a later transformation/analysis stage.
- Thresholds could be added to ensure that a single hue isn't parsed as different shades/tints due to real-world lighting or image quality considerations.

## Acknowledgements

- Example image was obtained and used under the Pixabay Content License
- Object detection model: [https://github.com/yainage90/fashion-visual-search]
- Segmentation model: Kirillov, Alexander, Mintun, Eric, Ravi, Nikhila, Mao, Hanzi, Rolland, Chloe, Gustafson, Laura, Xiao, Tete, Whitehead, Spencer, Berg, Alexander C., Lo, Wan-Yen, Dollár, Piotr, & Girshick, Ross. (2023). *Segment Anything*. arXiv:2304.02643. [https://arxiv.org/abs/2304.02643]
