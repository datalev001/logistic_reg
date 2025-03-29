####################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('cs-training.csv')
data.info()

'''
Data columns (total 15 columns):
 #   Column                                Non-Null Count   Dtype  
---  ------                                --------------   -----  
 0   Unnamed: 0                            150000 non-null  int64  
 1   SeriousDlqin2yrs                      150000 non-null  int64  
 2   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
 3   age                                   150000 non-null  int64  
 4   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
 5   DebtRatio                             150000 non-null  float64
 6   MonthlyIncome                         150000 non-null  float64
 7   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
 8   NumberOfTimes90DaysLate               150000 non-null  int64  
 9   NumberRealEstateLoansOrLines          150000 non-null  int64  
 10  NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
 11  NumberOfDependents                    150000 non-null  float64
 12  flag_MonthlyIncome                    150000 non-null  int32  
 13  flag_NumberOfDependents               150000 non-null  int32  
 14  cus_id                                150000 non-null  int64  

'''

#############################################
# Helper Functions
#############################################
def compute_ks(y_true, y_score):
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return 0.0
    df_temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    df_temp = df_temp.sort_values('y_score', ascending=False)
    df_temp['cum_positive'] = (df_temp['y_true'] == 1).cumsum() / pos
    df_temp['cum_negative'] = (df_temp['y_true'] == 0).cumsum() / neg
    ks_stat = max(abs(df_temp['cum_positive'] - df_temp['cum_negative']))
    return ks_stat

def acc_from_proba(y_true, y_score, threshold=0.5):
    pred_label = (y_score >= threshold).astype(int)
    return accuracy_score(y_true, pred_label)

def getcorr_cut(Y, df, varnamelist, thresh):
    """
    For each variable in varnamelist, compute correlation with Y.
    Keep those whose absolute correlation >= thresh.
    Return a DataFrame sorted by absolute correlation.
    """
    corr_records = []
    for vname in varnamelist:
        if vname not in df.columns:
            continue
        c = np.corrcoef(df[vname].fillna(0), Y)[0,1]
        corr_records.append((vname, c))
    corrdf = pd.DataFrame(corr_records, columns=['varname', 'correlation'])
    corrdf['abscorr'] = corrdf['correlation'].abs()
    corrdf = corrdf.query("abscorr >= @thresh").sort_values('abscorr', ascending=False)
    return corrdf

def select_top_features_for_AB(df_all: pd.DataFrame,
                               A_set: list,
                               B_dummy_cols: list,
                               top_n_A: int,
                               top_n_B: int,
                               target_col: str = 'badflag',
                               corr_threshold: float = 0.0):
    """
    From DataFrame df_all (which contains the target, A_set columns, and dummy columns from B set),
    select the top_n_A features from A_set and top_n_B features from B_dummy_cols based on absolute
    correlation with the target (>= corr_threshold). Returns three lists: A_top, B_top, and the combined final_features.
    """
    Y = df_all[target_col]
    # Top features from A_set
    A_corr = getcorr_cut(Y, df_all, A_set, corr_threshold)
    A_corr = A_corr.sort_values('abscorr', ascending=False).head(top_n_A)
    A_top = A_corr['varname'].tolist()
    
    # Top features from B_dummy_cols
    B_corr = getcorr_cut(Y, df_all, B_dummy_cols, corr_threshold)
    B_corr = B_corr.sort_values('abscorr', ascending=False).head(top_n_B)
    B_top = B_corr['varname'].tolist()
    
    final_features = A_top + B_top
    return A_top, B_top, final_features

#############################################
# 1) Load Data & Rename Columns
#############################################

# Rename long names to simple names
rename_map = {
    'SeriousDlqin2yrs': 'badflag',
    'RevolvingUtilizationOfUnsecuredLines': 'revol_util',
    'NumberOfTime30-59DaysPastDueNotWorse': 'pastdue_3059',
    'DebtRatio': 'debtratio',
    'MonthlyIncome': 'mincome',
    'NumberOfOpenCreditLinesAndLoans': 'opencredit',
    'NumberOfTimes90DaysLate': 'pastdue_90',
    'NumberRealEstateLoansOrLines': 'reloans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'pastdue_6089',
    'NumberOfDependents': 'numdep',
    'flag_MonthlyIncome': 'flag_mincome',
    'flag_NumberOfDependents': 'flag_numdep'
}
data = data.rename(columns=rename_map)

#############################################
# 2) Fill Missing & Create A set
#############################################
missing_vars = ['mincome', 'numdep']
vars2= ['revol_util', 'debtratio']
for mv in missing_vars:
    flagcol = 'flag_' + mv
    if flagcol not in data.columns:
        data[flagcol] = data[mv].isnull().astype(int)
data[missing_vars] = data[missing_vars].fillna(data[missing_vars].mean())
A_set = missing_vars + ['flag_' + m for m in missing_vars] + vars2

#############################################
# 3) Create B set by Dummies
#############################################
B_cats = ['pastdue_3059', 'pastdue_90',
          'reloans', 'pastdue_6089', 'opencredit']

dummy_frames = []
dummy_cols = []
for catvar in B_cats:
    if catvar not in data.columns:
        continue
    dums = pd.get_dummies(data[catvar], prefix=catvar)
    dummy_frames.append(dums)
    dummy_cols.extend(list(dums.columns))
B_dummies_df = pd.concat(dummy_frames, axis=1)

#############################################
# 4) Feature Selection: Correlation and Top N from A & B
#############################################
tmp_df = pd.concat([data[['badflag']], data[A_set], B_dummies_df], axis=1)
A_top, B_top, final_features = select_top_features_for_AB(
    df_all=tmp_df,
    A_set=A_set,
    B_dummy_cols=list(B_dummies_df.columns),
    top_n_A=8,
    top_n_B=17,
    target_col='badflag',
    corr_threshold=0.005
)
print("Top A-set features:", A_top)
print("Top B-set features:", B_top)
print("Final combined feature list:", final_features)

#############################################
# 5) Build Final Model DataFrame
#############################################
df_for_model = pd.concat([data[['badflag']], data[A_set], B_dummies_df], axis=1)
existing_feats = [f for f in final_features if f in df_for_model.columns]
model_df = df_for_model[['badflag'] + existing_feats]

#############################################
# 6) Train/Test Split
#############################################
train_df, test_df = train_test_split(
    model_df, test_size=0.25, random_state=12, stratify=model_df['badflag']
)
y_train = train_df['badflag']
y_test = test_df['badflag']
X_train = train_df.drop(columns=['badflag'])
X_test = test_df.drop(columns=['badflag'])

# save this data for future test
#train_df.to_csv('cs_X_train.csv', index = False)
#y_test.to_csv('cs_X_test.csv', index = False)

#############################################
# 7) Build Models: LightGBM, XGBoost, Logistic Regression
#############################################
# A) LightGBM
from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=4,
    random_state=42
)
lgb_clf.fit(X_train, y_train)
lgb_prob = lgb_clf.predict_proba(X_test)[:,1]
lgb_auc = roc_auc_score(y_test, lgb_prob)
lgb_ks = compute_ks(y_test, lgb_prob)
lgb_acc = acc_from_proba(y_test, lgb_prob, 0.5)
feat_imp_lgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_clf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# B) XGBoost
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.02,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)
xgb_prob = xgb_clf.predict_proba(X_test)[:,1]
xgb_auc = roc_auc_score(y_test, xgb_prob)
xgb_ks = compute_ks(y_test, xgb_prob)
xgb_acc = acc_from_proba(y_test, xgb_prob, 0.5)
feat_imp_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_clf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# C) Logistic Regression
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
log_clf.fit(X_train, y_train)
log_prob = log_clf.predict_proba(X_test)[:,1]
log_auc = roc_auc_score(y_test, log_prob)
log_ks = compute_ks(y_test, log_prob)
log_acc = acc_from_proba(y_test, log_prob, 0.5)

#############################################
# 8) Print Model Performance
#############################################
print("\n=== LightGBM Model Results ===")
print(f"AUC: {lgb_auc:.4f}, KS: {lgb_ks:.4f}, ACC: {lgb_acc:.4f}")
print("Feature Importances (LGB):")
print(feat_imp_lgb)

print("\n=== XGBoost Model Results ===")
print(f"AUC: {xgb_auc:.4f}, KS: {xgb_ks:.4f}, ACC: {xgb_acc:.4f}")
print("Feature Importances (XGB):")
print(feat_imp_xgb)

print("\n=== Logistic Regression Results ===")
print(f"AUC: {log_auc:.4f}, KS: {log_ks:.4f}, ACC: {log_acc:.4f}")
print("Logistic Regression does not provide built-in feature importances.")


'''
=== LightGBM Model Results ===
AUC: 0.8276, KS: 0.5197, ACC: 0.9366
Feature Importances (LGB):
            Feature  Importance
0         debtratio        3203
1           mincome        2502
2            numdep         701
3    pastdue_6089_0         586
4    pastdue_3059_0         529
5      pastdue_90_0         504
6      opencredit_0         317
7    pastdue_3059_1         301
8      pastdue_90_1         248
9    pastdue_6089_1         224
10   pastdue_3059_2         177
11     pastdue_90_2         159
12     pastdue_90_4         147
13   pastdue_3059_4         126
14      flag_numdep         122
15   pastdue_3059_3          99
16     flag_mincome          91
17   pastdue_6089_3          56
18     pastdue_90_3          34
19   pastdue_6089_2          29
20  pastdue_3059_98           7
21  pastdue_6089_98           0

=== XGBoost Model Results ===
AUC: 0.8291, KS: 0.5217, ACC: 0.9363
Feature Importances (XGB):
            Feature  Importance
0      pastdue_90_0    0.483000
1    pastdue_3059_0    0.228286
2    pastdue_6089_0    0.084587
3      pastdue_90_1    0.043893
4    pastdue_3059_1    0.039690
5      opencredit_0    0.018899
6      flag_mincome    0.012858
7    pastdue_6089_1    0.012383
8    pastdue_3059_2    0.012244
9         debtratio    0.009612
10          mincome    0.008938
11           numdep    0.007320
12   pastdue_3059_4    0.005876
13   pastdue_6089_3    0.005819
14      flag_numdep    0.005723
15     pastdue_90_2    0.005134
16     pastdue_90_4    0.004913
17   pastdue_3059_3    0.003407
18     pastdue_90_3    0.003051
19  pastdue_3059_98    0.002398
20   pastdue_6089_2    0.001970
21  pastdue_6089_98    0.000000

=== Logistic Regression Results ===
AUC: 0.8023, KS: 0.5011, ACC: 0.9347
Logistic Regression does not provide built-in feature importances.

'''