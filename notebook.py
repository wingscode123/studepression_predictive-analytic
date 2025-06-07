# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import plotly.express as px
import warnings  

# Ignore all warnings  
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
# %% [markdown]
# # Load Dataset

# %%
df = pd.read_csv('student_depression_dataset.csv')
df

# %%
df.info()

# %%
df.describe()

# %%
df.shape

# %% [markdown]
# # Data Cleaning & Preprocessing

# %% [markdown]
# ## Convert and Clean Data Types

# %%
# Drop kolom yang memiliki nilai 0 (kecuali kolom target & Work/Study Hours)
academic_pressure = (df['Academic Pressure'] == 0).sum()
work_pressure = (df['Work Pressure'] == 0).sum()
cgpa = (df['CGPA'] == 0).sum()
study_satisfaction = (df['Study Satisfaction'] == 0).sum()
job_satisfaction = (df['Job Satisfaction'] == 0).sum()

print("Nilai 0 di dalam kolom Academic Pressure ada: ", academic_pressure)
print("Nilai 0 di dalam kolom Work Pressure ada: ", work_pressure)
print("Nilai 0 di dalam kolom CGPA ada: ", cgpa)
print("Nilai 0 di dalam kolom Study Satisfaction ada: ", study_satisfaction)
print("Nilai 0 di dalam kolom Job Satisfaction ada: ", job_satisfaction)

# %%
# Drop kolom work pressure, id, Job Satisfaction
df.drop(columns=['Work Pressure','id', 'Job Satisfaction'], inplace=True)

# Hapus nilai 0 di kolom academic pressure, cgpa, dan Study Satisfaction
df = df.loc[(df[['Academic Pressure', 'CGPA', 'Study Satisfaction']]!=0).all(axis=1)]

# %%
# Validasi Ulang nilai kategori
df['Gender'] = df['Gender'].str.strip().str.lower().map({'male':'Male', 'female':'Female'})
df['Financial Stress'] = df['Financial Stress'].str.strip().str.title()

# %%
# Convert kolom object yang harus menjadi kategorikal
categorical_columns = ['Gender', 'Profession', 'Degree','Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Check unik value di beberapa kolom untuk menentukan cleaning strategy
print("Unique values in 'Sleep Duration':", df['Sleep Duration'].unique())
print("Unique values in 'Financial Stress':", df['Financial Stress'].unique())


# %%
# Kita buat function to extract numeric hours dari kolom 'Sleep Duration'
def extract_hours(s):
    # Find a numbers (include decimals)
    match = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(match.group(1)) if match else np.nan

df['Sleep Duration'] = df['Sleep Duration'].apply(extract_hours)

# Convert Financial Stress ke kategorikal if represent levels
df['Financial Stress'] = df['Financial Stress'].astype('category')

# Verify perubahan
print(df[['Sleep Duration', 'Financial Stress']].head())

# %% [markdown]
# ## Check for Missing Values

# %%
# display missing values per kolom
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# %%
# Ubah nilai missing values dengan median
for col in ['Sleep Duration']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# %% [markdown]
# ## Pengecekan Outliers

# %%
# Terapkan IQR
numeric_cols = df.select_dtypes(include='number').columns

Q1 =  df[numeric_cols].quantile(.25)
Q3 =  df[numeric_cols].quantile(.75)
IQR = Q3 - Q1

filter_outliers = ~((df[numeric_cols] < ( Q1 - 1.5 * IQR)) | (df[numeric_cols] > ( Q3 + 1.5 * IQR))).any(axis=1)
df = df[filter_outliers]
df.shape

# %%
df.info()

# %% [markdown]
# # EDA

# %%
# 1. Distribusi Target
plt.figure(figsize=(6,4))
sns.countplot(x='Depression', data=df, palette="viridis")
plt.title("Distribusi Depresi")
plt.xlabel("Depresi (0 = Tidak, 1 = Ya)")
plt.ylabel("Jumlah")
plt.show()
    
print("\nProporsi Kelas Target:")
print(df['Depression'].value_counts(normalize=True))


# %%
# 2. Kategorikal vs Target
cat_cols = ['Gender', 'Profession', 'Dietary Habits', 'Degree', 
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    
for col in cat_cols:
        plt.figure(figsize=(7,4))
        sns.countplot(x=col, hue='Depression', data=df, palette="Set2")
        plt.title(f"{col} vs {'Depression'}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Crosstab % analysis
        ct = pd.crosstab(df[col], df['Depression'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, colormap='Set3', figsize=(6,4))
        plt.title(f"Persentase {'Depression'} berdasarkan {col}")
        plt.ylabel("Persentase (%)")
        plt.tight_layout()
        plt.show()

# %%
# 3. Distribusi Fitur Numerik
num_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
            'Sleep Duration', 'Work/Study Hours']
    
df[num_cols].hist(bins=20, figsize=(14, 10), color='steelblue')
plt.suptitle("Distribusi Variabel Numerik", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# 4. Boxplot Numerik vs Target
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Depression', y=col, data=df, palette="pastel")
    plt.title(f"{col} berdasarkan {'Depression'}")
    plt.show()


# %%
 # 5. Korelasi Variabel Numerik
plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap Korelasi")
plt.show()

# %%
# 6. Interaktif: CGPA vs Study Satisfaction
if 'CGPA' in df.columns and 'Study Satisfaction' in df.columns:
    fig = px.scatter(df, x="CGPA", y="Study Satisfaction", 
                    color='Depression', 
                    hover_data=['Age', 'Gender', 'Academic Pressure'],
                    title="CGPA vs Study Satisfaction by Depression")
    fig.show()

# %%
# 7. Pairplot Visualisasi Interaksi Fitur
sns.pairplot(df[['CGPA', 'Study Satisfaction', 'Sleep Duration', 
                'Academic Pressure', 'Depression']], hue='Depression', palette="husl")
plt.suptitle("Pairplot Fitur Penting", y=1.02)
plt.show()

# %% [markdown]
# # Data Preparation

# %%
df

# %%
# Pisahkan fitur dan target
X = df.drop(columns=['Depression'])
y = df['Depression']

# Identifikasi tipe kolom
cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# %%
# Pipeline kategorikal: impute + one-hot encode
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline numerik: impute + scaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Gabungkan dengan ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# %%
# Transform data
X_processed = preprocessor.fit_transform(X)

# Dimensi hasil encoding
print(f"Shape setelah preprocessing: {X_processed.shape}")

# %%
# Cek explained variance per komponen
pca_check = PCA().fit(X_processed)
plt.plot(np.cumsum(pca_check.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid()
plt.show()

# %%
# PCA, misalnya ke 95% variansi
pca = PCA(n_components=25, random_state=42)
X_pca = pca.fit_transform(X_processed)

print(f"Shape setelah PCA: {X_pca.shape}")

# %%
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# %% [markdown]
# # Model Building

# %% [markdown]
# ## Logistic Regression Model

# %%
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

# Confusion matrix
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt="d", cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_prob_log = log_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression ROC curve (AUC = {roc_auc_log:.2f})', color='darkorange')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## Random Forest Classifier

# %%
# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix for RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve for RF
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest ROC curve (AUC = {roc_auc_rf:.2f})', color='green')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# # Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# %%
# Feature Importance (Random Forest)
importances = rf_model.feature_importances_
plt.figure(figsize=(10, 5))
plt.plot(importances)
plt.title("Feature Importances (PCA Components)")
plt.xlabel("Component Index")
plt.ylabel("Importance")
plt.show()

# %% [markdown]
# # Evaluasi Model

# %%
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Untuk Logistic Regression
ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test, cmap="Blues", display_labels=["No Depression", "Depression"])
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Untuk Random Forest
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap="Greens", display_labels=["No Depression", "Depression"])
plt.title("Confusion Matrix - Random Forest")
plt.show()


# %%
# Cross-validation for Logistic Regression
cv_scores_log = cross_val_score(log_model, X_pca, y, cv=5, scoring='accuracy')
print("5-Fold CV Accuracy for Logistic Regression:", cv_scores_log)
print("Mean CV Accuracy:", cv_scores_log.mean())

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X_pca, y, cv=5, scoring='accuracy')
print("5-Fold CV Accuracy for Random Forest:", cv_scores_rf)
print("Mean CV Accuracy:", cv_scores_rf.mean())

# %%
# Buat prediksi
def predict_depression(model, preprocessor, pca_model, raw_input):
    """
    Predict single input (dictionary) after preprocessing and PCA.
    """
    # Ubah dict jadi dataframe
    input_df = pd.DataFrame([raw_input])

    # Preprocess
    X_encoded = preprocessor.transform(input_df)
    X_pca = pca_model.transform(X_encoded)

    # Predict
    pred_class = model.predict(X_pca)[0]
    pred_prob = model.predict_proba(X_pca)[0][1]

    return {
        "prediction": int(pred_class),
        "confidence": float(pred_prob)
    }


sample_input = {
    "Age": 23,
    "Gender": "Male",
    "Sleep Duration": 6.5,
    "City" : "Nashik",
    "Profession": "Student",
    "Academic Pressure":5.75,
    "CGPA": 9.40,
    "Financial Stress": 5.0,
    "Study Satisfaction" : 2.0,
    "Dietary Habits": "Unhealthy",
    "Degree": "MSc",
    "Have you ever had suicidal thoughts ?": "Yes",
    "Work/Study Hours": 9.0,
    "Family History of Mental Illness": "Yes",
    
    
}

result = predict_depression(rf_model, preprocessor, pca, sample_input)
print("Prediction:", result['prediction'])
print("Confidence:", result['confidence'])


