# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# data collection
data = pd.read_csv(r"D:\python_projects\project 3\soil_cover_type.csv")
data

# data understanding
# **1 .checking shape of the dataframes :**
print ('data shape : ',data.shape)

# **2 .checking info of the dataframes :**
data.info()

# Separate numerical and categorical columns
data_nc = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
data_cc = data.select_dtypes(include=['object']).columns.tolist()
print(f"  Numerical columns ({len(data_nc)}): {data_nc}")
print(f"\n Categorical columns ({len(data_cc)}): {data_cc}")
print()

 # **3 .printing total missing values of the dataframes :**
data.isnull().sum()

# **4.checking datatypes of the dataframes :**
data.dtypes

# **5.checking duplicated values of the dataframes :**
data.duplicated().sum()

# **6.understanding the data using statistics :**
data.describe()
print("Unique values:")
for i, col in enumerate(data.columns):
    print(i, col,'-', data[col].nunique())
    
counts = data['Cover_Type'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.xlabel("Cover Type")
plt.ylabel("Count")
plt.xticks(rotation = 30)
plt.title("Cover Type Distribution")
plt.show()
print('\n Total Values:')
print(data['Cover_Type'].value_counts())

# **Check skewness of numerical columns**
print(data[data_nc].skew().sort_values(ascending=False))
plt.figure(figsize=(11,5))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - All Data')
plt.show()

# **Final summary of data understanding**
print(f"Total Rows: {data.shape[0]}")
print(f"Total Columns: {data.shape[1]}")
print(f"Numerical Columns: {len(data_nc)}")
print(f"Categorical Columns: {len(data_cc)}")
print(f"Total Missing Values: {data.isnull().sum().sum()}")
print(f"Total Duplicate Rows: {data.duplicated().sum()}\n")


# Data Cleaning
#  **1: Clean and rename column names :**
df_data = data.copy()
df_data  = df_data.rename(columns={
    'Horizontal_Distance_To_Hydrology':'Horizontal_Dist_To_water',
    'Vertical_Distance_To_Hydrology':'Vertical_Dist_To_water',
    'Horizontal_Distance_To_Fire_Points':'Horizontal_Dist_To_Fire',
})
df_data.columns = df_data.columns.str.lower().str.replace(' ', '_')
df_data['cover_type'] = df_data['cover_type'].str.strip().str.title()

print("df_data - Column names cleaned:")
print(df_data.columns.tolist())
print()

# **2: handling skewness :**
df_data_nc = df_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_data_cc = df_data.select_dtypes(include=['object']).columns.tolist()
df_data[df_data_nc].skew().sort_values()

from sklearn.preprocessing import PowerTransformer
sqrt_cols = ['aspect','slope','horizontal_dist_to_water']
for col in sqrt_cols:
    df_data[col] = np.sqrt(df_data[col])

yeo_cols = ['vertical_dist_to_water','hillshade_9am','hillshade_noon','elevation']
pt = PowerTransformer(method='yeo-johnson')
df_data[yeo_cols] = pt.fit_transform(df_data[yeo_cols])

#checking skewness after transformation:
print("Skewness after transformation:\n")
print(df_data[sqrt_cols + yeo_cols].skew())

# **3: handling outliers :**
numeric_cols = ['elevation','aspect','slope','horizontal_dist_to_water','vertical_dist_to_water',
    'horizontal_distance_to_roadways','hillshade_9am','hillshade_noon','hillshade_3pm',
    'horizontal_dist_to_fire']

for col in numeric_cols:
    Q1 = df_data[col].quantile(0.25)
    Q3 = df_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_data[(df_data[col] < lower) | (df_data[col] > upper)][col].count()
    print(f"{col} — Outliers: {outliers} ({round(outliers/len(df_data)*100, 2)}%)")
print()

#EDA
# Histogram — distribution of all numerical columns
plt.figure(figsize=(10,10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 3, i)  
    sns.histplot(df_data[col], bins=30, kde=True, color='steelblue', edgecolor='black')
    plt.title(col)

plt.suptitle('Histogram  — All Numerical Columns \n', fontsize=16)
plt.tight_layout()
plt.show()

# boxplot — outlier of all numerical columns
plt.figure(figsize=(10,10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 3, i)  
    sns.boxplot(df_data[col])
    plt.title(col)

plt.suptitle('Boxplot  — All Numerical Columns \n', fontsize=16)
plt.tight_layout()
plt.show()

# Countplot — frequency of each categorical column
df_data_cc = ['cover_type' ,'wilderness_area']
plt.figure(figsize=(10, 5))
for i, col in enumerate(df_data_cc):
    plt.subplot(1, 2, i+1)
    sns.countplot(x=df_data[col], hue=df_data[col], palette='colorblind', legend=False)
    plt.title(col)
    plt.xticks(rotation=45)
plt.suptitle('Countplot — All Categorical Columns \n', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df_data, x='soil_type', color='steelblue')
plt.title('Soil Type')
plt.show()

# Pie Chart — proportion of target variable
plt.figure(figsize=(9, 9))
df_data['cover_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['steelblue', 'salmon','red','green','yellow'])
plt.title('Pie Chart — Target variable  Distribution')
plt.ylabel('')
plt.show()

# Correlation Heatmap — relationships between all numerical columns
plt.figure(figsize=(12, 8))
sns.heatmap(df_data[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap — All Numerical Columns')
plt.tight_layout()
plt.show()

# Feature engineering
# Check highly correlated features above 0.8
corr_matrix = df_data[numeric_cols].corr().abs()

for col in corr_matrix.columns:
    for row in corr_matrix.index:
        if corr_matrix.loc[row, col] > 0.8 and row != col:
            print(f"{col} — {row} : {corr_matrix.loc[row, col].round(3)}")

# total_distance
df_data['total_distance'] = (
    df_data['horizontal_dist_to_fire'] +
    df_data['horizontal_distance_to_roadways'] +
    df_data['horizontal_dist_to_water']
)

# Average shade
df_data['avg_hillshade'] = (
    df_data['hillshade_9am'] +
    df_data['hillshade_noon'] +
    df_data['hillshade_3pm']
) / 3

# Water distance
df_data['water_distance'] = np.sqrt(
    df_data['horizontal_dist_to_water']**2 +
    df_data['vertical_dist_to_water']**2
)

# Shade variation
df_data['shade_range'] = (
    df_data[['hillshade_9am',
             'hillshade_noon',
             'hillshade_3pm']].max(axis=1)
    -
    df_data[['hillshade_9am',
             'hillshade_noon',
             'hillshade_3pm']].min(axis=1)
)

# Elevation × slope
df_data['elevation_slope'] = (
    df_data['elevation'] *
    df_data['slope']
)

#drop columns with high collinearity:
df_data = df_data.drop(columns=['hillshade_3pm'])

# checking for class imbalance
print (df_data['cover_type'].value_counts(normalize=True) * 100 , '\n')

print(df_data['soil_type'].value_counts(normalize=True).sort_index() * 100 ,'\n')

df_data['wilderness_area'].value_counts(normalize=True) * 100

# Train test split
from sklearn.model_selection import train_test_split
x = df_data.drop('cover_type', axis=1)
y = df_data['cover_type']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify= y)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"\n x_train: {x_train.shape}, x_test: {x_test.shape}")

# **encoding**
categorical_cols = ['soil_type', 'wilderness_area']

# Encode training data
x_train_ohe = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True )

# Encode test data
x_test_ohe = pd.get_dummies( x_test, columns=categorical_cols, drop_first=True )

# Align columns
x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join='left', axis=1, fill_value=0 )

# Verify
print(f"\n x_train_ohe shape: {x_train_ohe.shape}, x_test_ohe shape: {x_test_ohe.shape}")
print(f"\nNew Columns Added:\n\n x_train_ohe\n :{x_train_ohe.columns.tolist()}, \n\n x_test_ohe :\n {x_test_ohe.columns.tolist()} ")
print("\nOneHot Encoding Done! ")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"\ny_train_encoded shape: {y_train_encoded.shape}")
print(f"y_test_encoded shape: {y_test_encoded.shape}")

print("\nEncoded Classes:")
print(le.classes_)
print("\nTarget Encoding Done!")

# Convert bool columns to int
bool_cols = x_train_ohe.select_dtypes(include='bool').columns
x_train_ohe[bool_cols] = x_train_ohe[bool_cols].astype(int)

# Convert category dtype to int
cat_dtype_cols = x_train_ohe.select_dtypes(include='category').columns
x_train_ohe[cat_dtype_cols] = x_train_ohe[cat_dtype_cols].astype(int)

# Verify
print(x_train_ohe.dtypes)
print(f"\nAny bool columns remaining: {x_train_ohe.select_dtypes(include='bool').columns.tolist()}")
print(f"Any category dtype remaining: {x_train_ohe.select_dtypes(include='category').columns.tolist()}")
print("\nAll fixed! ")

# **scaling**
scaling_cols = ['elevation', 'aspect', 'slope', 'horizontal_dist_to_water',
       'vertical_dist_to_water', 'horizontal_distance_to_roadways',
       'hillshade_9am', 'hillshade_noon', 'horizontal_dist_to_fire','total_distance', 'avg_hillshade',
       'water_distance', 'shade_range', 'elevation_slope']



from sklearn.preprocessing import StandardScaler

# Copy datasets
x_train_scaled = x_train_ohe.copy()
x_test_scaled = x_test_ohe.copy()

scaler = StandardScaler()
x_train_scaled[scaling_cols] = scaler.fit_transform(x_train_ohe[scaling_cols])
x_test_scaled[scaling_cols] = scaler.transform(x_test_ohe[scaling_cols])

print("Scaling Done! ")

import pickle
# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# Save column names
with open('columns.pkl', 'wb') as f:
    pickle.dump(x_train_scaled.columns.tolist(), f)
print("Files saved successfully ")

# **Handling class imbalance**
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample( x_train_scaled,  y_train_encoded )
print('smote done')

#verify befire and after smote
print('Before smote: \n',pd.Series(y_train_encoded).value_counts()) 
print('\n after smote: \n',pd.Series(y_train_smote).value_counts())

with open('x_train_smote.pkl', 'wb') as f:
    pickle.dump(x_train_smote, f)

with open('y_train_smote.pkl', 'wb') as f:
    pickle.dump(y_train_smote, f)

with open('x_test_scaled.pkl', 'wb') as f:
    pickle.dump(x_test_scaled, f)

with open('y_test_encoded.pkl', 'wb') as f:
    pickle.dump(y_test_encoded, f)
print("Files saved ")

import joblib
joblib.dump(scaler, "scaler.pkl")
# ----------------------xxxxx----------------------------------


with open('x_train_smote.pkl', 'rb') as f:
    x_train_smote = pickle.load(f)

with open('y_train_smote.pkl', 'rb') as f:
    y_train_smote = pickle.load(f)

with open('x_test_scaled.pkl', 'rb') as f:
    x_test_scaled = pickle.load(f)

with open('y_test_encoded.pkl','rb') as f:
    y_test_encoded = pickle.load(f)

print("Files loaded ")

# Model Building

# **Random forest**
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


rfc_model = RandomForestClassifier( n_estimators=100, n_jobs=-1,random_state=42 )
rfc_model.fit(x_train_smote, y_train_smote)
rfc_predictions = rfc_model.predict(x_test_scaled)

rf_accuracy = accuracy_score(y_test_encoded, rfc_predictions)
rf_Precision = precision_score(y_test_encoded, rfc_predictions, average='weighted')
rf_Recall = recall_score(y_test_encoded, rfc_predictions, average='weighted')
rf_F1_Score = f1_score(y_test_encoded, rfc_predictions, average='weighted')


print(f"rf_accuracy: {rf_accuracy:.2f}")
print(f"rf_Precision: {rf_Precision:.2f}")
print(f"rf_Recall: {rf_Recall:.2f}")
print(f"rf_F1_Score: {rf_F1_Score:.2f}")

from sklearn.metrics import confusion_matrix
print('\n',confusion_matrix(y_test_encoded, rfc_predictions))

# **DecisionTreeClassifier**
dtc_accuracy = accuracy_score(y_test_encoded, Dtc_prediction)
dtc_Precision = precision_score(y_test_encoded, Dtc_prediction, average='weighted')
dtc_Recall = recall_score(y_test_encoded, Dtc_prediction, average='weighted')
dtc_F1_Score = f1_score(y_test_encoded, Dtc_prediction, average='weighted')


print(f"dtc_accuracy: {dtc_accuracy:.2f}")
print(f"dtc_Precision: {dtc_Precision:.2f}")
print(f"dtc_Recall: {dtc_Recall:.2f}")
print(f"dtc_F1_Score: {dtc_F1_Score:.2f}")

from sklearn.metrics import confusion_matrix
print('\n confusion matrix \n',confusion_matrix(y_test_encoded, rfc_predictions))

# **logistic regression**
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(x_train_smote, y_train_smote)
lr_predictions = lr_model.predict(x_test_scaled)
print('model trained')

from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test_encoded, lr_predictions)}")
print('\n',classification_report(y_test_encoded, lr_predictions))

lr_accuracy = accuracy_score(y_test_encoded, lr_predictions)
lr_Precision = precision_score(y_test_encoded, lr_predictions, average='weighted')
lr_Recall = recall_score(y_test_encoded, lr_predictions, average='weighted')
lr_F1_Score = f1_score(y_test_encoded, lr_predictions, average='weighted')


print(f"lr_accuracy: {lr_accuracy:.2f}")
print(f"lr_Precision: {lr_Precision:.2f}")
print(f"lr_Recall: {lr_Recall:.2f}")
print(f"lr_F1_Score: {lr_F1_Score:.2f}")

# **XG Boost**
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42)

xgb_model.fit(x_train_smote, y_train_smote)

xgb_predictions = xgb_model.predict(x_test_scaled)
print('model trained')

xgb_accuracy = accuracy_score(y_test_encoded, xgb_predictions)
xgb_Precision = precision_score(y_test_encoded, xgb_predictions,average='weighted')
xgb_Recall = recall_score(y_test_encoded, xgb_predictions, average='weighted')
xgb_F1_Score = f1_score(y_test_encoded, xgb_predictions, average='weighted')


print(f"xgb_accuracy: {xgb_accuracy:.2f}")
print(f"xgb_Precision: {xgb_Precision:.2f}")
print(f"xgb_Recall: {xgb_Recall:.2f}")
print(f"xgb_F1_Score: {xgb_F1_Score:.2f}")

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train_smote, y_train_smote)
knn_predictions = knn.predict(x_test_scaled)
print('model trained')

knn_accuracy = accuracy_score(y_test_encoded, knn_predictions)
knn_Precision = precision_score(y_test_encoded, knn_predictions,average='weighted')
knn_Recall = recall_score(y_test_encoded, knn_predictions, average='weighted')
knn_F1_Score = f1_score(y_test_encoded, knn_predictions, average='weighted')

print(f"knn_accuracy: {knn_accuracy:.2f}")
print(f"knn_Precision: {knn_Precision:.2f}")
print(f"knn_Recall: {knn_Recall:.2f}")
print(f"knn_F1_Score: {knn_F1_Score:.2f}")

# checking final score

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Decision Tree",
        "Logistic Regression",
        "XGBoost",
        "KNN"
    ],
    "Accuracy": [
        rf_accuracy,
        dtc_accuracy,
        lr_accuracy,
        xgb_accuracy,
        knn_accuracy
    ],
    "F1": [
        rf_F1_Score,
        dtc_F1_Score,
        lr_F1_Score,
        xgb_F1_Score,
        knn_F1_Score
    ]
})
print(results.sort_values("F1", ascending=False))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    rfc_model,
    x_train_smote,
    y_train_smote,
    cv=5,
    scoring='f1_weighted'
)

print(scores)
print(scores.mean())

# train vs test accuracy
train_pred = rfc_model.predict(x_train_smote)
print(
    "Train Accuracy:",
    accuracy_score(y_train_smote, train_pred)
)
print(
    "Test Accuracy:",
    accuracy_score(y_test_encoded, rfc_predictions)
)

# checking random test cases
import random
for _ in range(10):
    idx = random.randint(0, len(x_test_scaled)-1)

    actual = y_test_encoded.iloc[idx] if hasattr(y_test_encoded, 'iloc') else y_test_encoded[idx]

    pred = rfc_model.predict(
        x_test_scaled.iloc[[idx]] if hasattr(x_test_scaled, 'iloc')
        else x_test_scaled[idx:idx+1]
    )[0]

    print(f"Index: {idx}")
    print(f"Actual: {actual}")
    print(f"Predicted: {pred}")
    print()

# finding best parameters
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=5,          
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    random_state=42
)

random_search.fit(x_train_smote, y_train_smote)

print(random_search.best_params_)
print(random_search.best_score_)

# tuning
best_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

best_rf.fit(x_train_smote, y_train_smote)

pred = best_rf.predict(x_test_scaled)

print("Accuracy:", accuracy_score(y_test_encoded, pred))
print("F1:", f1_score(y_test_encoded, pred, average='weighted'))

# saving best model
import pickle
with open("random_forest.pkl", "wb") as f:
    pickle.dump(best_rf, f)

print('file loaded')

with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)
print('done')

# -----------------------xxxxx----------------------------