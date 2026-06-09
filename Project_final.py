# ============================================
# FRAUD DETECTION PROJECT — FINAL SOLUTION
# PaySim Dataset
# Main model: Random Forest
# ============================================

# This project predicts whether a transaction is fraud or normal.
# We use Random Forest as the final main model.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# This helps us get the same results when we run the code again.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Change this only if your dataset file has a different name.
DATA_PATH = "PS_20174392719_1491204439457_log.csv"

# Output folders.
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_output_folders():
    """Create folders for results and saved model."""
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def clean_old_results():
    """Remove old result files, so unwanted old plots do not stay there."""
    for file_path in RESULTS_DIR.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def save_plot(file_name):
    """Save the current plot into the results folder."""
    output_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def load_dataset(data_path):
    """Load the PaySim dataset."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}\n"
            "Put the PaySim CSV file in the same folder as this script, "
            "or change DATA_PATH in the code."
        )

    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    return df


def print_basic_information(df):
    """Print simple dataset information."""
    print("\n========== BASIC DATA INFORMATION ==========")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["isFraud"].value_counts())

    fraud_ratio = df["isFraud"].mean() * 100
    print(f"\nFraud ratio: {fraud_ratio:.4f}%")


# ============================================
# EXPLORATORY DATA ANALYSIS PLOTS
# ============================================

def fraud_distribution_log_scale(df):
    """
    Show normal and fraud transaction counts.

    The normal fraud distribution plot is not useful because fraud cases are very small.
    So here we use log scale to make both classes visible.
    """
    counts = df["isFraud"].value_counts().sort_index()
    total = len(df)

    plot_df = pd.DataFrame({
        "Class": ["Normal", "Fraud"],
        "Count": [counts.get(0, 0), counts.get(1, 0)],
    })

    plot_df["Percentage"] = plot_df["Count"] / total * 100
    plot_df.to_csv(RESULTS_DIR / "fraud_distribution_table.csv", index=False)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="Class", y="Count")

    plt.yscale("log")
    plt.title("Fraud Distribution using Log Scale")
    plt.xlabel("Transaction class")
    plt.ylabel("Number of transactions log scale")

    # Add count and percentage labels.
    for i, row in plot_df.iterrows():
        ax.text(
            i,
            row["Count"],
            f'{int(row["Count"]):,}\n({row["Percentage"]:.4f}%)',
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_plot("fraud_distribution_log_scale.png")


def transaction_type_analysis(df):
    """
    Show transaction type vs fraud.

    This is similar to the old plot, but uses log scale so the fraud bars are visible.
    """
    plt.figure(figsize=(9, 5))
    sns.countplot(x="type", hue="isFraud", data=df)

    plt.yscale("log")
    plt.title("Transaction Type vs Fraud using Log Scale")
    plt.xlabel("Transaction type")
    plt.ylabel("Number of transactions log scale")
    plt.xticks(rotation=45)
    plt.legend(title="isFraud", labels=["Normal", "Fraud"])

    save_plot("transaction_type_vs_fraud_log_scale.png")

    # Save a clear table for the report.
    type_table = df.groupby("type").agg(
        total_transactions=("isFraud", "count"),
        fraud_cases=("isFraud", "sum"),
    )

    type_table["fraud_rate_percent"] = (
        type_table["fraud_cases"] / type_table["total_transactions"] * 100
    )

    type_table = type_table.sort_values(
        by="fraud_rate_percent",
        ascending=False,
    )

    type_table.to_csv(RESULTS_DIR / "transaction_type_fraud_summary.csv")

    print("\nTransaction type fraud summary:")
    print(type_table)


def amount_distribution(df):
    """
    Show transaction amount distribution.

    We keep this old plot because it is useful.
    We use log scale because transaction amounts have a large range.
    """
    plot_df = df

    # Use sample only for plotting speed. Model still uses full data.
    if len(df) > 200000:
        plot_df = df.sample(n=200000, random_state=RANDOM_STATE)
        print("Using 200000 rows sample for amount plot.")

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=plot_df,
        x="amount",
        hue="isFraud",
        bins=100,
        log_scale=True,
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction amount log scale")
    plt.ylabel("Number of transactions")
    plt.legend(title="isFraud", labels=["Fraud", "Normal"])

    save_plot("amount_distribution.png")


def create_eda_plots(df):
    """Create simple and useful EDA plots."""
    print("\n========== CREATING EDA PLOTS ==========")

    fraud_distribution_log_scale(df)
    transaction_type_analysis(df)
    amount_distribution(df)


# ============================================
# DATA PREPARATION
# ============================================

def prepare_data(df):
    """Prepare the data before model training."""
    df = df.copy()

    # New feature: money removed from the sender account.
    df["balanceOrigDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    # New feature: money added to the receiver account.
    df["balanceDestDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # These columns are not useful for prediction.
    # isFlaggedFraud is removed because it is a system flag.
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert transaction type text into numeric columns.
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # X contains input features. y contains the target value.
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # Stratify keeps the fraud ratio similar in train and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, X.columns


# ============================================
# MODEL TRAINING
# ============================================

def train_random_forest(X_train, y_train):
    """Train the Random Forest model."""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)
    return rf_model


# ============================================
# MODEL EVALUATION
# ============================================

def evaluate_random_forest(model, X_test, y_test):
    """Evaluate the Random Forest model using important metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n========== RANDOM FOREST CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Extra useful metrics for imbalanced fraud detection.
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    metrics = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
    }

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"],
    )

    metrics_df.to_csv(RESULTS_DIR / "random_forest_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"],
    )

    cm_df.to_csv(RESULTS_DIR / "random_forest_confusion_matrix.csv")

    # Save full classification report as a table.
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    report_df.to_csv(RESULTS_DIR / "random_forest_classification_report.csv")

    print("\nFinal metrics table:")
    print(metrics_df)

    return y_pred, y_prob, cm, metrics_df


def random_forest_confusion_matrix_plot(cm):
    """
    Plot the confusion matrix.

    We show both counts and row percentages.
    This makes the result easier to understand.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums * 100

    labels = np.array([
        [
            f"{cm[0, 0]:,}\n{cm_percent[0, 0]:.2f}%",
            f"{cm[0, 1]:,}\n{cm_percent[0, 1]:.2f}%",
        ],
        [
            f"{cm[1, 0]:,}\n{cm_percent[1, 0]:.2f}%",
            f"{cm[1, 1]:,}\n{cm_percent[1, 1]:.2f}%",
        ],
    ])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_percent,
        annot=labels,
        fmt="",
        cmap="Greens",
        xticklabels=["Predicted Normal", "Predicted Fraud"],
        yticklabels=["Actual Normal", "Actual Fraud"],
        cbar_kws={"label": "Percentage inside each actual class"},
    )

    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")

    save_plot("random_forest_confusion_matrix.png")


def random_forest_roc_curve_plot(y_test, y_prob):
    """Plot ROC curve for Random Forest."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curve - Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    save_plot("random_forest_roc_curve.png")


def random_forest_precision_recall_curve_plot(y_test, y_prob):
    """
    Plot Precision-Recall curve.

    This is useful because the dataset is highly imbalanced.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")

    plt.title("Precision-Recall Curve - Random Forest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    save_plot("random_forest_precision_recall_curve.png")


def random_forest_feature_importance_plot(model, feature_names):
    """Plot the most important features used by Random Forest."""
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False,
    )

    feature_importance.to_csv(
        RESULTS_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    print("\nTop 15 feature importances:")
    print(feature_importance.head(15))

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance.head(15),
    )

    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    save_plot("random_forest_feature_importance.png")


def create_evaluation_plots(model, y_test, y_prob, cm, feature_names):
    """Create final model evaluation plots."""
    print("\n========== CREATING MODEL EVALUATION PLOTS ==========")

    random_forest_confusion_matrix_plot(cm)
    random_forest_roc_curve_plot(y_test, y_prob)
    random_forest_precision_recall_curve_plot(y_test, y_prob)
    random_forest_feature_importance_plot(model, feature_names)


# ============================================
# MAIN PROGRAM
# ============================================

def main():
    """Run the full final project pipeline."""
    create_output_folders()
    clean_old_results()

    df = load_dataset(DATA_PATH)
    print_basic_information(df)

    create_eda_plots(df)

    print("\n========== PREPARING DATA ==========")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\n========== TRAINING RANDOM FOREST ==========")
    rf_model = train_random_forest(X_train, y_train)

    print("\n========== EVALUATING RANDOM FOREST ==========")
    y_pred, y_prob, cm, metrics_df = evaluate_random_forest(
        rf_model,
        X_test,
        y_test,
    )

    create_evaluation_plots(
        rf_model,
        y_test,
        y_prob,
        cm,
        feature_names,
    )

    # Save the trained model.
    model_path = MODELS_DIR / "random_forest_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"\nSaved model: {model_path}")

    print("\nProject run finished successfully.")
    print("Check the 'results' folder for final plots and tables.")


if __name__ == "__main__":
    main()