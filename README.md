# cextract: Content Extraction

This repository provides a set of tools for structurally analyzing HTML and applying a machine learning model to identify the main article content.

The core component, cextract.py, utilizes a Keras-based multi-input model to predict the probability that any given DOM element is part of the main article content, filtering out navigational, structural, and advertisement elements.

## Pipeline

The extraction process involves the following steps:

1.  DOM Flattening: An input HTML document is converted into a flat CSV-like structure using Depth-First Search (DFS), where relationships are maintained via explicit ID referencing (`{idXXX}`).
2.  Structural Preprocessing: The DOM elements are transformed into sequences (XPath, attributes) and numerical features (depth, text length). This step uses the `DOMPreprocessor` class, which ensures consistency with the training data.
3.  Content Prediction: A pre-trained Keras model analyzes the processed features to output a probability score for each element being part of the main content.
4.  Content Building: A post-processing step (`content_builder.build`) aggregates the high-scoring elements and reconstructs the clean article text.

## Prerequisites
  * `pandas`
  * `beautifulsoup4`
  * `tensorflow` (CPU recommended, as configured in the script)
  * `numpy`
  * `scikit-learn` (for `StandardScaler`)
  * `pickle`

## Setup

Before running the main script, two crucial artifacts must be placed in the `models/` directory:

1.  `dom_predictor_model.h5`: The pre-trained Keras model file.
2.  `dom_preprocessor.pkl`: The pickled instance of `DOMPreprocessor` containing the fitted tokenizers and scaler.

The required local modules (`download_html`, `html2csv`, `content_builder`) must also be available in the project environment.

## Usage

Run the main extraction script from the command line, providing the target URL as an argument.

```bash
python cextract.py "http://example.com/article/123"
```

### Script Arguments

| Argument | Description |
| :--- | :--- |
| `<URL>` | The full URL of the webpage to analyze. |
