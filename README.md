# Fraud

Fraud is a project that explores `classification techniques` in the context of `artificial intelligence`.
The dataset used is `Credit Card Fraud Detection Dataset 2023`, from Kaggle, which "(...) contains credit card transactions made by European cardholders in the year 2023 (...)".

#### :test_tube: Work
The original dataset is balanced, i.e., contains the same proportion of fraudulent observations as non-fraudulent observations. To make it more challenging and closer to a real-world scenario, the **proportion of fraudulent observations was set to 1%**. The `main.py` notebook aims to show how one could create an intelligent system to accurately identify both fraudulent and non-fraudulent transactions.

Note that, in this case, a *random classifier which predicts fraudulent 1% of time and non-fraudulent 99% of the time*, would already achieve an *accuracy of 98.02%*. If this serves as a baseline model, then the model to be developed needs to perform better, i.e., an accuracy greater than 98.02%.

```python
Accuracy = P(Forecast = Actual)
         = P(A=1) x P(F=1 | A=1) + P(A=0) x P(F=0 | A=0)
         = P(A=1) x P(F=1)       + P(A=0) x P(F=0)
         = 0.01^2                + 0.99^2

# Similarly, it can be shown that precision, recall and f1-score are equal to 0.01, for this random classifier. 
```

The notebooks folder, explores other material such as decision boundaries in 2D or threshold optimization through `predict_proba` method.

<p align="center">
  <img src="https://github.com/user-attachments/assets/302f7113-d606-420a-9582-5d16b5a38b44" />
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
