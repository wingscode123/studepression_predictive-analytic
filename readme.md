# Laporan Proyek Machine Learning - Radithya Fawwaz Aydin

## Domain Proyek

Kesehatan mental mahasiswa telah menjadi isu kritis yang memerlukan perhatian serius dari institusi pendidikan tinggi. Depresi di kalangan mahasiswa tidak hanya berdampak pada performa akademik, tetapi juga pada motivasi, kesejahteraan psikologis, dan kualitas hidup secara menyeluruh. Menurut data dari berbagai penelitian, prevalensi depresi di kalangan mahasiswa terus meningkat dan memerlukan sistem deteksi dini yang efektif.

Masalah ini harus diselesaikan karena dampak depresi yang tidak tertangani dapat berlanjut hingga mahasiswa mengalami penurunan prestasi akademik, putus kuliah, bahkan dalam kasus ekstrem dapat mengarah pada pikiran atau tindakan bunuh diri. Sistem deteksi dini berbasis data dapat membantu institusi pendidikan untuk memberikan intervensi yang tepat waktu kepada mahasiswa yang memerlukan bantuan.

Dengan memanfaatkan teknologi machine learning, kita dapat menganalisis pola dari data demografis, akademik, dan gaya hidup mahasiswa untuk mengidentifikasi faktor-faktor risiko dan membangun model prediktif yang akurat. Hal ini memungkinkan implementasi sistem peringatan dini yang proaktif dalam lingkungan kampus.

**Referensi:**
- Utami, N. (2021). Pengaruh Kesehatan Mental terhadap Prestasi Akademik Mahasiswa. *Jurnal Psikologi Pendidikan*, 12(3), 145-160.
- Sari, D., Pratiwi, R., & Wibowo, A. (2022). Efektivitas Intervensi Berbasis Data untuk Deteksi Dini Depresi pada Mahasiswa. *Jurnal Kesehatan Mental Mahasiswa*, 7(2), 89-104.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, berikut adalah pernyataan masalah yang akan diselesaikan dalam proyek ini:

1. **Bagaimana memanfaatkan data demografis, akademik, dan gaya hidup untuk memprediksi risiko depresi pada mahasiswa?**
   - Diperlukan identifikasi fitur-fitur yang paling berpengaruh terhadap kondisi depresi mahasiswa dari berbagai aspek kehidupan mereka.

2. **Algoritma klasifikasi apa yang memberikan performa terbaik dalam mendeteksi mahasiswa dengan gejala depresi?**
   - Perlu dilakukan perbandingan antar algoritma untuk menentukan model yang paling efektif dalam konteks deteksi dini depresi.

3. **Bagaimana membangun sistem prediksi yang dapat diimplementasikan untuk intervensi preventif di lingkungan kampus?**
   - Diperlukan model yang tidak hanya akurat tetapi juga dapat diinterpretasi dan diimplementasikan dalam sistem informasi kampus.

### Goals

Tujuan yang ingin dicapai dari proyek ini adalah:

1. **Membangun model klasifikasi biner yang dapat memprediksi status depresi mahasiswa** (`Depression`: 1 = depresi, 0 = tidak depresi) berdasarkan data demografis, akademik, dan gaya hidup dengan tingkat akurasi yang tinggi.

2. **Mengidentifikasi algoritma machine learning terbaik** melalui perbandingan performa antara Logistic Regression dan Random Forest Classifier menggunakan metrik evaluasi yang relevan.

3. **Menganalisis dan mengidentifikasi fitur-fitur yang paling signifikan** dalam memprediksi kondisi depresi mahasiswa untuk memberikan insight yang actionable bagi institusi pendidikan.

### Solution Statements

Untuk mencapai goals yang telah ditetapkan, berikut adalah solusi yang akan diimplementasikan:

1. **Pengembangan dua model klasifikasi yang berbeda:**
   - **Logistic Regression**: Sebagai baseline model yang memberikan interpretabilitas tinggi dan cocok untuk masalah klasifikasi biner.
   - **Random Forest Classifier**: Sebagai ensemble model yang dapat menangani data dengan fitur yang kompleks dan memberikan feature importance.

2. **Implementasi teknik preprocessing dan feature engineering yang komprehensif:**
   - Penanganan missing values, encoding kategorikal, feature scaling, dan dimensionality reduction dengan PCA.
   - Evaluasi dampak setiap teknik preprocessing terhadap performa model.

3. **Evaluasi model menggunakan multiple metrics:**
   - Menggunakan precision, recall, F1-score, dan accuracy untuk evaluasi yang komprehensif.
   - Fokus khusus pada recall untuk kelas depresi (class 1) karena pentingnya mendeteksi kasus positif dalam konteks kesehatan mental.

Semua solusi ini dapat terukur melalui metrik evaluasi yang telah ditentukan dan akan memberikan hasil yang dapat diimplementasikan dalam sistem nyata.

## Data Understanding

Dataset yang digunakan dalam proyek ini diambil dari Kaggle dengan judul **"Student Depression Dataset: Analyzing Mental Health Trends and Predictors Among Students"**. Dataset ini dapat diakses melalui tautan berikut: [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset).

Dataset ini disusun untuk menganalisis tren dan faktor-faktor prediktif dari depresi di kalangan mahasiswa. Setiap baris dalam dataset merepresentasikan satu responden mahasiswa dengan berbagai atribut terkait kondisi demografis, akademik, dan kesejahteraan psikologis.

### Karakteristik Dataset

**📊 Dimensi Data:**
- **Jumlah Baris**: 27.901 sampel mahasiswa
- **Jumlah Kolom**: 17 variabel (16 fitur + 1 target variable)
- **Ukuran Dataset**: Cukup besar untuk training model yang robust

**🔍 Kualitas Data:**
- **Missing Values**: Beberapa kolom memiliki nilai kosong/null yang memerlukan treatment khusus
- **Data Cleaning**: Diperlukan pembersihan data untuk menangani missing values dan inconsistency
- **Outliers**: Terdapat outlier pada beberapa variabel numerik yang perlu dianalisis lebih lanjut
- **Data Types**: Mix antara categorical, numerical, dan ordinal variables

**🧹 Strategi Data Cleaning:**
- **Imputation**: Menggunakan strategi berbeda untuk numerical (mean) dan categorical (most frequent)
- **Outlier Handling**: Identifikasi dan treatment outlier untuk mencegah bias model
- **Feature Engineering**: Preprocessing untuk categorical encoding dan numerical scaling

### Variabel-variabel pada Student Depression Dataset

**Fitur Demografis:**
- `ID`: Identifier unik untuk setiap responden
- `Age`: Usia mahasiswa (dalam tahun)
- `Gender`: Jenis kelamin (Male/Female)
- `City`: Kota asal mahasiswa

**Fitur Akademik:**
- `CGPA`: Cumulative Grade Point Average (IPK mahasiswa)
- `Academic Pressure`: Tingkat tekanan akademik yang dirasakan (skala numerik)
- `Study Satisfaction`: Tingkat kepuasan terhadap studi (skala numerik)
- `Degree`: Jenjang pendidikan yang sedang ditempuh

**Fitur Gaya Hidup dan Kesejahteraan:**
- `Sleep Duration`: Durasi tidur per hari (dalam jam)
- `Dietary Habits`: Kebiasaan makan (Healthy/Moderate/Unhealthy)
- `Work/Study Hours`: Jam kerja/belajar per hari
- `Profession`: Profesi atau bidang pekerjaan

**Fitur Faktor Risiko:**
- `Financial Stress`: Tingkat stres finansial (skala numerik)
- `Family History of Mental Illness`: Riwayat keluarga terkait masalah mental health (Yes/No)
- `Have you ever had suicidal thoughts ?`: Adanya pikiran bunuh diri (Yes/No)

**Target Variable:**
- `Depression`: Status depresi (1 = depresi, 0 = tidak depresi)

### Pentingnya Data Quality Assessment

Analisis awal terhadap kualitas data sangat penting untuk:
- **Memahami** karakteristik dan keterbatasan dataset
- **Menentukan** strategi preprocessing yang tepat
- **Mengidentifikasi** potensi bias atau masalah dalam data
- **Memastikan** reliability dan validity dari model yang dibangun

### Exploratory Data Analysis (EDA)

Untuk memahami karakteristik data secara mendalam, dilakukan beberapa tahapan analisis eksplorasi:

**1. Analisis Distribusi Kelas**
- Visualisasi proporsi antara mahasiswa yang mengalami depresi vs tidak mengalami depresi
- Identifikasi adanya class imbalance dan strategi untuk mengatasinya

**2. Analisis Demografis**
- Histogram distribusi usia mahasiswa untuk memahami sebaran usia responden
- Pie chart komposisi gender untuk melihat representasi gender dalam dataset
- Bar chart distribusi kota asal untuk memahami sebaran geografis

**3. Analisis Korelasi**
- Heatmap korelasi antar fitur numerik (CGPA, Sleep Duration, Work/Study Hours, Financial Stress) terhadap target variable
- Identifikasi multikolinearitas antar variabel prediktor menggunakan correlation matrix
- Analisis hubungan antara fitur-fitur penting dengan target variable

**4. Deteksi Outlier dan Missing Values**
- Boxplot untuk mendeteksi outlier pada fitur numerik seperti `Sleep Duration`, `CGPA`, dan `Work/Study Hours`
- Visualisasi missing value matrix menggunakan heatmap untuk mengevaluasi completeness data
- Statistik deskriptif untuk memahami distribusi setiap fitur

**5. Analisis Bivariat**
- Perbandingan distribusi CGPA dan Sleep Duration berdasarkan status depresi menggunakan violin plot
- Stacked bar chart untuk menganalisis hubungan antara tekanan akademik, stres finansial, dan pikiran bunuh diri terhadap status depresi
- Cross-tabulation analysis untuk fitur kategorikal terhadap target variable

Melalui EDA ini, diperoleh pemahaman yang komprehensif tentang pola dan karakteristik data, serta validasi bahwa dataset layak untuk diproses lebih lanjut dalam tahap preprocessing dan pemodelan.

## Data Preparation

Tahapan data preparation dilakukan secara sistematis untuk memastikan kualitas data optimal. Berikut langkah-langkah yang dilakukan:

### 1. Data Cleaning & Type Conversion
- Konversi tipe data numerik (`Age`, `CGPA`, `Sleep Duration`) dari object ke numeric
- Drop kolom `ID` karena tidak memiliki nilai prediktif
- Standardisasi format data kategorikal

### 2. Missing Values & Outlier Handling
- **Imputasi median** untuk `Sleep Duration` (distribusi skewed, robust terhadap outlier)
- **Imputasi mean** untuk fitur numerik lainnya dengan distribusi normal
- **Imputasi most frequent** untuk fitur kategorikal
- **Outlier removal** menggunakan metode IQR pada fitur numerik utama

### 3. Preprocessing Pipeline
```python
# Pipeline kategorikal: impute → one-hot encode
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline numerik: impute → scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
```

**Alasan:** Pipeline memastikan konsistensi preprocessing dan mencegah data leakage.

### 4. Dimensionality Reduction (PCA)
- Aplikasi PCA setelah scaling untuk menghindari bias skala
- Seleksi **25 komponen** berdasarkan explained variance analysis
- **Tujuan:** Mengurangi dimensi, mengatasi multikolinearitas, mencegah overfitting

### 5. Train-Test Split
- **80% training, 20% testing** dengan stratified split
- Mempertahankan proporsi kelas target yang seimbang
- `random_state=42` untuk reproducibility

### Hasil Akhir
- **Shape setelah preprocessing:** [27,901 samples, encoded_features]
- **Shape setelah PCA:** [27,901 samples, 25 components]  
- **Kualitas:** No missing values, scaled features, optimal dimensionality

Dataset siap untuk modeling dengan transformasi yang sistematis dan terintegrasi.
## Modeling

Pada tahap ini, dua algoritma machine learning diterapkan untuk menyelesaikan masalah klasifikasi status depresi mahasiswa: **Logistic Regression** dan **Random Forest Classifier**. Model dilatih menggunakan dataset terpisah (`X_train`, `y_train`).

---

### Logistic Regression


**Deskripsi Singkat:**
Model klasifikasi linear yang memetakan probabilitas kelas menggunakan fungsi sigmoid.

**Parameter yang digunakan:**
- `random_state=42` – untuk memastikan reproducibility
- `max_iter=1000` – menambah jumlah iterasi agar model dapat konvergen
- `solver='lbfgs'` *(default)* – solver efisien untuk dataset ukuran sedang, mendukung L2-regularization

---

### Random Forest Classifier

**Deskripsi Singkat:**
Model ensemble berbasis decision tree yang menggabungkan banyak pohon untuk meningkatkan generalisasi.

**Parameter yang digunakan:**
- `n_estimators=100` – jumlah pohon dalam forest
- `random_state=42` – untuk hasil yang konsisten

---

### ⚙️ Implementasi Model

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```


## Evaluasi Model Prediksi Depresi Mahasiswa

### Metrik Evaluasi

### 1. Accuracy
**Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Mengukur proporsi prediksi yang benar dari total prediksi.

### 2. Precision
**Formula:** `Precision = TP / (TP + FP)`

Proporsi prediksi positif yang benar - penting untuk mengurangi false alarm dalam konteks kesehatan mental.

### 3. Recall (Sensitivity)
**Formula:** `Recall = TP / (TP + FN)`

Proporsi kasus depresi yang berhasil dideteksi - **metrik paling krusial** untuk mencegah missed detection.

### 4. F1-Score
**Formula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

Harmonic mean dari precision dan recall.

### 5. AUC-ROC
Area Under the Curve - mengukur kemampuan model membedakan kelas pada berbagai threshold.

## Hasil Evaluasi

### Logistic Regression
```
              precision    recall  f1-score   support
           0       0.84      0.79      0.81      2310
           1       0.86      0.89      0.87      3265

    accuracy                           0.85      5575
   macro avg       0.85      0.84      0.84      5575
weighted avg       0.85      0.85      0.85      5575
```
**AUC-ROC:** 0.92

### Random Forest
```
              precision    recall  f1-score   support
           0       0.82      0.80      0.81      2310
           1       0.86      0.88      0.87      3265

    accuracy                           0.85      5575
   macro avg       0.84      0.84      0.84      5575
weighted avg       0.85      0.85      0.85      5575
```
**AUC-ROC:** 0.91

### Confusion Matrix Summary

**Logistic Regression:**
- True Negative: 1,825 | False Positive: 485
- False Negative: 355 | True Positive: 2,910

**Random Forest:**
- True Negative: 1,853 | False Positive: 457  
- False Negative: 395 | True Positive: 2,870

### Cross-Validation Results
- **Logistic Regression CV Score:** 0.847 ± 0.012
- **Random Forest CV Score:** 0.844 ± 0.015

## Analisis Performa

### Keunggulan Logistic Regression
1. **Recall untuk kelas depresi: 0.89 vs 0.88** - Mendeteksi 33 kasus depresi lebih banyak dibanding Random Forest
2. **AUC-ROC lebih tinggi: 0.92 vs 0.91** - Kemampuan diskriminasi yang superior
3. **Stabilitas cross-validation lebih baik** - Standard deviation lebih rendah (0.012 vs 0.015)

### Implikasi Perbedaan Kecil
Meskipun perbedaan recall hanya 0.01 (1%), dalam konteks kesehatan mental hal ini berarti:
- **33 mahasiswa depresi tambahan terdeteksi** dari 3,265 kasus
- Potensi intervensi dini untuk mencegah perburukan kondisi
- Cost-effectiveness lebih tinggi dalam alokasi sumber daya konseling

## Kesimpulan

**Logistic Regression dipilih sebagai model optimal** berdasarkan:

1. **Recall tertinggi (0.89)** - Prioritas utama dalam deteksi kesehatan mental
2. **AUC-ROC superior (0.92)** - Kemampuan diskriminasi terbaik
3. **Konsistensi cross-validation** - Performa stabil across different data splits
4. **Interpretabilitas tinggi** - Memungkinkan analisis faktor risiko yang mendalam

Model ini dapat diimplementasikan sebagai sistem peringatan dini dengan kemampuan mendeteksi 89% kasus depresi mahasiswa, memberikan foundation solid untuk intervensi kesehatan mental di lingkungan kampus.

---

**Catatan:** Laporan ini menunjukkan bahwa implementasi machine learning dalam deteksi dini depresi mahasiswa memiliki potensi besar untuk diterapkan dalam lingkungan pendidikan tinggi sebagai bagian dari sistem monitoring kesehatan mental yang komprehensif.