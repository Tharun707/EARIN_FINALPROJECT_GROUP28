# Fraud Detection in Financial Transactions

This project uses the PaySim dataset to detect fraudulent financial transactions.
The final model is a Random Forest classifier.

## Dataset

Download the PaySim dataset from Kaggle and place the CSV file in the same folder as `Project_final.py`.

Expected CSV file name:

```text
PS_20174392719_1491204439457_log.csv
```

If your file has a different name, change the `DATA_PATH` variable inside `Project_final.py`.

## How to run the project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python file:

```bash
python Project_final.py
```

## Output files

After running the code, the following folders will be created:

```text
results/
models/
```

The `results/` folder contains:

- fraud distribution plot
- transaction type vs fraud plot
- amount distribution plot
- Random Forest confusion matrix
- Random Forest ROC curve
- Random Forest feature importance plot
- final Random Forest metrics table

The `models/` folder contains:

- trained Random Forest model saved as `random_forest_model.pkl`

## Final model

Random Forest was selected as the main model because it performed well during the midterm phase and can capture non-linear patterns in the dataset.
It also provides feature importance, which helps explain which transaction features are most useful for fraud detection.
