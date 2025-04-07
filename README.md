# Fraud

Fraud is a project that explores `classification techniques` in the context of `artificial intelligence`.
The dataset used is `Credit Card Fraud Detection Dataset 2023`, from Kaggle, which "(...) contains credit card transactions made by European cardholders in the year 2023 (...)".

#### :file_folder: Repository structure
```python
  ├── notebooks/                # exploration notebooks
  | ├── decision_boundary.ipynb
  | └── threshold.ipynb
  ├── src/
  │ ├── data/                   # data loading and preprocessing utilities
  │ │ ├── prepare_data.py
  │ │ └── utils.py
  │ ├── models/                 # model-related components
  │ │ ├── classification.py
  │ │ ├── evaluation.py
  │ │ ├── feature_selection.py  # statistical tests and wrapper methods
  │ │ ├── model_selection.py    # grid-search and threshold evaluation
  │ │ └── utils.py
  │ └── visuals/
  │   ├── boundary.py           # 2D decision boundaries
  │   └── pr_roc_curve.py       # precision-recall and roc curves
  ├── .gitignore
  ├── README.md
  ├── config.yml                # configuration file with parameters and settings
  └── main.ipynb
  ```

#### :rocket: Installation

Follow the steps below to set up this project on your local machine.
Open the terminal and run the following lines of code.

```bash
# 1. Navigate to the directory where you'd like to save the project
>> cd /path/to/your/preferred/location

# 2. Clone the repo by running
>> git clone https://github.com/6oncvlo/Fraud.git

# 3. Navigate into the project folder
>> cd Fraud

# 4. [Optional] Set up and activate virtual environment
>> python -m venv .venv  
>> .venv\Scripts\activate

# 5. Install project dependencies
>> pip install -r requirements.txt

```
#### :handshake: References
- [Kaggle Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
