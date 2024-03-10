
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp



data = pd.read_csv('cs-training.csv')
data.info()

###################fitting logistic regression model#######################
missing_vars = ['MonthlyIncome', 'NumberOfDependents']
data['flag_MonthlyIncome'] = data.MonthlyIncome.isnull().astype(int)
data['flag_NumberOfDependents'] = data.NumberOfDependents.isnull().astype(int)
data[missing_vars] = data[missing_vars].fillna(data[missing_vars].mean())
missing_vars_clean = missing_vars + ['flag_' + name for name in missing_vars]
data['cus_id'] = range(1, len(data) + 1)  # add an id column with sequential integers
data[missing_vars_clean].head()

data.isnull().sum()
cols = list(data.columns)
cols.remove('SeriousDlqin2yrs')

X_train, X_test, y_train, y_test = train_test_split(data[cols], data['SeriousDlqin2yrs'], test_size=0.3, random_state=11)

logist_model = LogisticRegression(random_state=0).fit(X_train, y_train)

proba_pred = logist_model.predict_proba(X_test[cols])[:,1]
X_test['predict_prob'] = list(proba_pred)
pred = logist_model.predict(X_test[cols])
X_test['predict_vote'] = list(pred)

accuracy_score(y_test, X_test['predict_vote'])

feature_names = cols
intercept = logist_model.intercept_[0]
coefficients = logist_model.coef_[0]  

#############coeffcients p values##################################
# Add a constant to the predictors to account for the intercept
X = sm.add_constant(data[cols])
# The target variable
y = data['SeriousDlqin2yrs']
# Fit the model
model = sm.Logit(y, X).fit()
# Print the summary of the model to get the p-values
print(model.summary())

# Assuming 'data' is your DataFrame and 'target' is the name of the target variable
X = data.drop('SeriousDlqin2yrs', axis=1)  # Predictor variables
y = data['SeriousDlqin2yrs']  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)
standardized_coefficients = model.coef_[0]
coefficients_df = pd.DataFrame({'Variable': X.columns, 'Standardized Coefficient': standardized_coefficients})
print(coefficients_df)

###############################################
precision_score(y_test, X_test['predict_vote'])
###############################################

cutoffs = np.linspace(0, 1, 11)  # Creates 10 decile points from 0 to 1
precision_scores = []
for cutoff in cutoffs:
    pred_cutoff = np.where(proba_preds >= cutoff, 1, 0)
    precision = precision_score(y_test, pred_cutoff)
    precision_scores.append(precision)

plt.figure(figsize=(10, 6))
plt.plot(cutoffs, precision_scores, marker='o')
plt.title('Precision Score vs. Probability Cutoff')
plt.xlabel('Probability Cutoff Points')
plt.ylabel('Precision Score')
plt.grid(True)
plt.show()

#################AUC ROC##############################
# Assuming y_test are your true labels and proba_preds are the predicted probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, proba_preds)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Print AUC value
print(f"AUC (Area Under The Curve): {roc_auc:.2f}")


###############KS################################

# Assume y_true is your array of true labels (0s and 1s), and proba_preds are the predicted probabilities for the positive class
positive_probs = proba_preds[y_test == 1]
negative_probs = proba_preds[y_test == 0]

# Calculate the KS statistic (distance) between the positive and negative distributions
ks_statistic, p_value = ks_2samp(positive_probs, negative_probs)

print(f"KS Statistic (Distance): {ks_statistic}")


###############################################
fpr, tpr, thresholds = roc_curve(y_test, proba_preds)

ks_distance = max(tpr - fpr)
ks_index = np.argmax(tpr - fpr)  # Index where KS distance is maximum

x_ks = fpr[ks_index]
y_ks = tpr[ks_index]
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o', linestyle='-', color='b', label='TPR (Cumulative Gains)')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='No Skill Line')
plt.axvline(x=x_ks, color='red', linestyle='--', label=f'KS Distance = {ks_distance:.2f}')
plt.annotate(f'KS Distance\n{x_ks:.2f}, {y_ks:.2f}', xy=(x_ks, y_ks / 2), xytext=(x_ks + 0.1, y_ks / 2),
             arrowprops=dict(facecolor='red', shrink=0.05), horizontalalignment='right')

plt.title('Cumulative Gains Chart with KS Distance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()

