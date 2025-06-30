Markdown

# Bristol House Price Prediction

## Project Overview

This project focuses on building and evaluating machine learning models to forecast house prices in Bristol, UK. It leverages historical real estate data, performing data cleaning, feature engineering, and training various regression models (Linear Regression, Random Forest, XGBoost) to predict future property values.

## Key Features

* **Data Preprocessing:** Scripts to clean raw house price data, extract relevant features (e.g., postcode prefixes, time-based features), and prepare datasets for modeling.
* **Multiple Models:** Implements and compares the performance of:
    * Linear Regression
    * Random Forest Regressor
    * XGBoost Regressor (with hyperparameter tuning)
* **Model Evaluation:** Uses RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) to assess prediction accuracy on unseen 2024 data.
* **Feature Importance:** Visualizes feature importance for the XGBoost model.
* **Model Saving:** Saves trained models for future use.

Okay, here is the project structure again, formatted for easy copy-pasting.

**Project Structure:**

```
bristol_house_price_prediction/
├── data/
│   ├── bristol_houseprice_dataset_2021.csv.xlsx  # Example raw input data
│   ├── bristol_houseprice_dataset_2024.csv.xlsx  # Example raw input data
│   ├── bristol_2021_2023_model_data.csv.xlsx    # Processed training data
│   └── bristol_2024_model_data.csv.xlsx       # Processed test data
├── notebooks/                  # (Optional) Jupyter notebooks for EDA or experimentation
├── models/                     # Stores trained machine learning models (.pkl files)
├── plots/                      # Stores generated plots (e.g., feature importance, residuals)
├── scripts/                    # Contains all Python source code for data prep and model training
│   ├── fixed_dataset.py        # Script to process 2024 data (and similar for other years)
│   ├── fixed_dataset_1.py      # Script for initial raw data filtering (e.g., 2021 full UK data to Bristol)
│   ├── train_linear_forecast.py# Trains and evaluates Linear Regression model
│   ├── train_rf_forecast.py    # Trains and evaluates Random Forest model
│   └── train_xgb_model.py      # Trains and evaluates XGBoost model (with tuning)
├── .gitignore                  # Specifies intentionally untracked files
├── README.md                   # Project overview and instructions
└── requirements.txt            # List of Python dependencies
```


## Setup Instructions

### Prerequisites

* Python 3.x
* A working internet connection to download dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/bristol_house_price_prediction.git](https://github.com/YOUR_GITHUB_USERNAME/bristol_house_price_prediction.git)
    cd bristol_house_price_prediction
    ```
    *(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Follow these steps in sequence to reproduce the data processing, model training, and evaluation:

1.  **Prepare Raw Data:**
    * Place your raw house price datasets (e.g., `bristol_houseprice_dataset_2021.csv.xlsx`, `bristol_houseprice_dataset_2024.csv.xlsx`, etc., which are used as input for `fixed_dataset.py`) into the `data/` directory.
    * If you are starting from a full UK dataset (like `pp-2021.csv` mentioned in `fixed_dataset_1.py`), first place that file (or similar raw data for other years) into `data/`.
    * Then, run `fixed_dataset_1.py` (and similar scripts if you adapt them for other years) to extract Bristol-specific data and save it into the `data/` folder.
        ```bash
        python scripts/fixed_dataset_1.py
        # If you have fixed_dataset.py adapted for 2021-2023 data:
        # python scripts/fixed_dataset_for_2021_2023.py
        python scripts/fixed_dataset.py # This one is for 2024 data
        ```
    * **Ensure that `bristol_2021_2023_model_data.csv.xlsx` and `bristol_2024_model_data.csv.xlsx` are generated and present in your `data/` folder before proceeding.** These are the files your training scripts expect.

2.  **Train and Evaluate Models:**
    Run each training script from the project's root directory:
    ```bash
    python scripts/train_linear_forecast.py
    python scripts/train_rf_forecast.py
    python scripts/train_xgb_model.py
    ```
    These scripts will print evaluation metrics to the console, save trained models to the `models/` directory, and save plots to the `plots/` directory.
