# Fraud

Fraud is a project that explores `classification techniques` in the context of `artificial intelligence` to perform fraud detection.
The dataset used is [`Credit Card Fraud Detection (UBL)`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), from Kaggle, which "(...) contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. (...)".

#### :test_tube: Work
The `main.py` notebook aims to show how one could create an intelligent system to accurately identify both fraudulent and non-fraudulent transactions.

Using a baseline model that *randomly predicts fraud 0.173% of the time and non-fraud 99.827% of the time would already yield an accuracy of 99.65%.* Therefore, the model being developed must surpass this accuracy.

```python
Accuracy = P(Forecast = Actual)
         = P(A=1) x P(F=1 | A=1) + P(A=0) x P(F=0 | A=0)
         = P(A=1) x P(F=1)       + P(A=0) x P(F=0)
         = (0.173%)^2            + (1-0.173%)^2

# Similarly, it can be shown that precision, recall and f1-score are equal to 0.173%, for this random classifier. 
```

The notebooks folder, explores other material such as decision boundaries in 2D or threshold optimization through `predict_proba` method.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b3dee06d-b155-42b2-b0d4-092c5c941ac5" />
</p>
<p align="center"><em>Figure 1:</em> Example of decision boundary for logistic regression algorithm.</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f7157b4c-2509-4fdc-91aa-6b5c296c9f92" />
</p>
<p align="center"><em>Figure 2:</em> Example of classification threshold optimization.</p>

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
  ├── main.ipynb
  └── creditcard.csv            # To be added
  ```

**Note:** The path location of the dataset is stored in the `config.yml` file. Please adjust it or upload the dataset to your local repository.

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
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
