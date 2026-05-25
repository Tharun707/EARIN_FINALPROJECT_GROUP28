# ============================================
# FRAUD DETECTION PROJECT — MIDTERM PIPELINE
# PaySim Dataset
# ============================================


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	roc_auc_score,
	roc_curve,
	precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# CHANGE PATH IF NECESSARY
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

print(df.shape)
df.head()

df.info()
df.isnull().sum()

# Basic Data Analaysis
print(df['isFraud'].value_counts())
fraud_ratio = df['isFraud'].mean() * 100
print(f"Fraud Ratio: {fraud_ratio:.4f}%")

def fraud_distribution_plot(plt):
	plt.figure(figsize=(10,6))
	sns.countplot(x='isFraud', data=df)
	plt.title("Fraud Distribution")
	plt.xlabel("Fraud")
	plt.ylabel("Count")
	plt.show()

def transaction_type_analysis(plt):
	plt.figure(figsize=(10,6))
	sns.countplot(x='type', hue='isFraud', data=df)
	plt.title("Transaction Types vs Fraud")
	plt.xticks(rotation=45)
	plt.show()
	fraud_by_type = pd.crosstab(df['type'], df['isFraud'], normalize='index') * 100
	print(fraud_by_type)
	
def correlation_heatmap(plt):
	plt.figure(figsize=(12,8))
	numeric_df = df.select_dtypes(include=np.number)
	sns.heatmap(
		numeric_df.corr(),
		cmap='coolwarm',
		annot=False
	)
	plt.title("Correlation Heatmap")
	plt.show()
	
def amount_distribution(plt):
	plt.figure(figsize=(10,5))
	sns.histplot(
		data=df,
		x='amount',
		hue='isFraud',
		bins=100,
		log_scale=True
	)
	plt.title("Transaction Amount Distribution")
	plt.show()
	
	
# FEATURE ENGINEERING

def prepare_data(df):
	
	# Feature engineering
	df['balanceOrigDiff'] = (
			df['oldbalanceOrg'] - df['newbalanceOrig']
	)
	
	df['balanceDestDiff'] = (
			df['newbalanceDest'] - df['oldbalanceDest']
	)
	
	# Drop unnecessary columns
	drop_cols = [
		'nameOrig',
		'nameDest',
		'isFlaggedFraud'
	]
	
	df = df.drop(columns=drop_cols)
	
	# One-hot encoding
	df = pd.get_dummies(
		df,
		columns=['type'],
		drop_first=True
	)
	
	# Features / target
	X = df.drop('isFraud', axis=1)
	y = df['isFraud']
	
	# Split
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=y
	)
	
	return X_train, X_test, y_train, y_test, X.columns

# LOGISTIC REG BASELINE MODEL

def log_reg(x_train, y_train, x_test, y_test):
	
	scaler = StandardScaler()
	
	X_train_scaled = scaler.fit_transform(x_train)
	X_test_scaled = scaler.transform(x_test)
	
	log_model = LogisticRegression(
		class_weight='balanced',
		random_state=RANDOM_STATE,
		max_iter=1000
	)
	
	log_model.fit(X_train_scaled, y_train)
	
	y_pred_log = log_model.predict(X_test_scaled)
	
	y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
	
	print(classification_report(y_test, y_pred_log))
	
	# Confusion Matrix
	cm = confusion_matrix(y_test, y_pred_log)
	
	plt.figure(figsize=(6,5))
	
	sns.heatmap(
		cm,
		annot=True,
		fmt='d',
		cmap='Blues'
	)
	
	plt.title("Logistic Regression Confusion Matrix")
	
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	
	plt.show()
	
	# ROC Curve
	fpr, tpr, _ = roc_curve(y_test, y_prob_log)
	
	auc_score = roc_auc_score(y_test, y_prob_log)
	
	plt.figure(figsize=(7,5))
	
	plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
	
	plt.plot([0,1], [0,1], linestyle='--')
	
	plt.title("ROC Curve - Logistic Regression")
	
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	
	plt.legend()
	
	plt.show()
# MAIN MODEL
def rand_forest(
		x_train,
		y_train,
		x_test,
		y_test,
		feature_names
):
	
	rf_model = RandomForestClassifier(
		n_estimators=100,
		max_depth=10,
		class_weight='balanced',
		random_state=RANDOM_STATE,
		n_jobs=-1
	)
	
	rf_model.fit(x_train, y_train)
	
	y_pred_rf = rf_model.predict(x_test)
	
	y_prob_rf = rf_model.predict_proba(x_test)[:,1]
	
	print(classification_report(y_test, y_pred_rf))
	
	# Confusion Matrix
	cm_rf = confusion_matrix(y_test, y_pred_rf)
	
	plt.figure(figsize=(6,5))
	
	sns.heatmap(
		cm_rf,
		annot=True,
		fmt='d',
		cmap='Greens'
	)
	
	plt.title("Random Forest Confusion Matrix")
	
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	
	plt.show()
	
	# ROC
	fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
	
	roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
	
	plt.figure(figsize=(7,5))
	
	plt.plot(
		fpr_rf,
		tpr_rf,
		label=f"AUC = {roc_auc_rf:.4f}"
	)
	
	plt.plot([0,1], [0,1], linestyle='--')
	
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	
	plt.title("ROC Curve - Random Forest")
	
	plt.legend()
	
	plt.show()
	
	# Feature Importance
	feature_importance = pd.DataFrame({
		'Feature': feature_names,
		'Importance': rf_model.feature_importances_
	})
	
	feature_importance = feature_importance.sort_values(
		by='Importance',
		ascending=False
	)
	
	print(feature_importance.head(15))
	
	plt.figure(figsize=(10,6))
	
	sns.barplot(
		x='Importance',
		y='Feature',
		data=feature_importance.head(15)
	)
	
	plt.title("Top 15 Feature Importances")
	
	plt.show()
	
# MAIN

fraud_distribution_plot(plt)

transaction_type_analysis(plt)

correlation_heatmap(plt)

amount_distribution(plt)

X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

print("========== Logistic Regression ==========")

log_reg(X_train, y_train, X_test, y_test)

print("========== Random Forest ==========")

rand_forest(
	X_train,
	y_train,
	X_test,
	y_test,
	feature_names
)