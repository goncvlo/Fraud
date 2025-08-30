# Fraud

Fraud is a project that explores `classification techniques` in the context of `artificial intelligence` to perform fraud detection.
The dataset used is [`Credit Card Fraud Detection (UBL)`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), from Kaggle, which "(...) contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. (...)".

### :test_tube: Work
The `main.py` notebook aims to show how one could create an intelligent system to accurately identify both fraudulent and non-fraudulent transactions.

#### Baseline Model
Using as baseline model an estimator which *randomly predicts fraud 0.173% of the time and non-fraud 99.827% of the time, would already yield an accuracy of 99.65%.* In such an extreme case, note that an *estimator which always "predicts" non-fraud, achieves an accuracy of 99.827% - as it correctly identifies all the non-fradulent transactions but misses to correctly identify all the fraudelent transactions.*

In either case, the model being developed must surpass this accuracy.

```python
# accuracy computation for 1st baseline model
Accuracy = P(Forecast = Actual)
         = P(A=1) x P(F=1 | A=1) + P(A=0) x P(F=0 | A=0)
         = P(A=1) x P(F=1)       + P(A=0) x P(F=0)
         = (0.173%)^2            + (1-0.173%)^2

# similarly, it can be shown that precision, recall and f1-score are equal to 0.173%
```

#### Model

The table below presents performance metrics from cross-validation evaluations of various algorithms. Each algorithm was optimized for accuracy using Bayesian optimization (Optuna, n_trials = 10) with oversampling applied during training. Check `model_selection.ipynb` notebook for more details.

| Algorithm   | Accuracy | Precision | Recall | F1 Score | Trial Index | Training Time (secs) |
|-------------|----------|-----------|--------|----------|----------|----------|
| LGBMClassifier :trophy: | 0.99978 | 0.95455 | 0.91304 | 0.93333 | 0 | 8.1 |
| RandomForestClassifier | 0.99963 | 0.90909 | 0.86957 | 0.88889 | 2 | 24.6 |
| XGBClassifier | 0.99956 | 0.86957 | 0.86957 | 0.86957 | 0 | 5.7 |
| NeuralNetworkClassifier | 0.99941 | 0.80000 | 0.86957 | 0.83333 | 7 | 201.1 |
| DecisionTreeClassifier :no_entry: | 0.99727 | 0.36538 | 0.82609 | 0.50667 | 6 | 19.1 |
| LogisticRegression :no_entry: | 0.97413 | 0.05930 | 0.95652 | 0.11168 | 2 | 1.5 |

Note that
- all four metrics - accuracy, precision, recall and f1-score, have range in $[0, 1]$.
- algorithms tagged with the :no_entry: emoji, don't reach the baseline model accuracy.
- a model with 100% accuracy represents a 0.17% improvement over the baseline model. LGBMClassifier has an improvement of 0.15%.
- final evaluation should be done in the test set. Therefore, accuracy on test set for best model is 0.99951, which represents an improvement of 0.12%.

The notebooks folder explores additional topics, including `2D decision boundaries`, `deep learning modeling`, `threshold optimization`, and `bootstrap technique`.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b43ed4dc-7e2f-4f69-aeab-b6f5406e9fcf" />
</p>
<p align="center"><em>Figure 1:</em> Example of decision boundary for logistic regression algorithm.</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f8f03615-faac-45c8-84fd-2a4f8d1f0274" />
</p>
<p align="center"><em>Figure 2:</em> Example of classification threshold optimization.</p>

### :file_folder: Repository structure
```python
  ├── notebooks/                # exploration notebooks
  | ├── bootstrap.ipynb         # ensemble model
  | ├── decision_boundary.ipynb
  | ├── deep_learning.ipynb
  | ├── feature_selection.ipynb
  | ├── model_selection.ipynb
  | └── threshold.ipynb
  ├── src/
  │ ├── data/                   # data loading and preprocessing utilities
  │ │ ├── prepare_data.py
  │ │ └── utils.py
  │ ├── models/                 # model-related components
  │ │ ├── evaluation.py
  │ │ ├── feature_selection.py  # statistical tests and wrapper methods
  │ │ ├── model.py
  │ │ ├── tracker.py            # mlflow for experiment tracking
  │ │ ├── tuner.py              # bayesian search for hyperparam and sample_weight optimization
  │ │ └── utils.py
  │ └── visuals/
  │   ├── boundary.py           # 2D decision boundaries
  │   └── pr_roc_curve.py       # precision-recall and roc curves
  ├── .gitignore
  ├── config.yml                # configuration file with parameters and settings
  ├── creditcard.csv            # to be added
  ├── main.ipynb
  ├── README.md
  └── requirements.txt          # project dependencies
  ```

**Note:** The path location of the dataset is stored in the `config.yml` file. Please adjust it or upload the dataset to your local repository.

### :handshake: References
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Optuna](https://optuna.org/)
- [MLflow](https://mlflow.org/)
- [Keras](https://www.tensorflow.org/guide/keras)
  
