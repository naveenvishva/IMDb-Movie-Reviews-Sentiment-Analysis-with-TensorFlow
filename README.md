
# IMDb Movie Reviews Sentiment Analysis with TensorFlow

Perform sentiment analysis on IMDb movie reviews using TensorFlow.

![IMDb Movie Reviews](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/IMDB_Logo_2016.svg/1200px-IMDB_Logo_2016.svg.png)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Author](#author)
- [License](#license)

## Overview

This project utilizes TensorFlow to perform sentiment analysis on IMDb movie reviews. It includes scripts for training a neural network model on the IMDb dataset (`train_and_save_model.py`) and evaluating the model's performance (`load_and_evaluate_model.py`). Additionally, there's a `requirements.txt` file listing the dependencies required for running the scripts.

## Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your_username/IMDb-Movie-Reviews-Sentiment-Analysis.git
   ```

2. Navigate to the project directory:
   ```
   cd IMDb-Movie-Reviews-Sentiment-Analysis
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the model and save it:
   ```
   python train_and_save_model.py
   ```

2. Evaluate the model's performance and visualize the confusion matrix:
   ```
   python load_and_evaluate_model.py
   ```

## File Descriptions

1. **train_and_save_model.py**: Trains a neural network model on IMDb movie reviews dataset and saves the trained model as `moviereview.h5`.

2. **load_and_evaluate_model.py**: Loads the saved model `moviereview.h5`, evaluates its performance on test data, generates predictions, creates a confusion matrix, and visualizes the matrix as a heatmap.

3. **requirements.txt**: Lists required Python packages and their versions.

4. **moviereview.h5**: Saved trained model file.

## Requirements

- Python 3.x
- TensorFlow 2.6.0
- Matplotlib 3.4.3
- Seaborn 0.11.2

## Author

[Your Name/Username]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
