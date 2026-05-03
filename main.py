import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv('data/dataset_tugas1_preprocessing.csv')

# ======================
# 2. EDA
# ======================
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# ======================
# 3. HANDLE MISSING VALUES
# ======================
df['Umur'] = df['Umur'].fillna(df['Umur'].mean())
df['Nilai_Akhir'] = df['Nilai_Akhir'].fillna(df['Nilai_Akhir'].mode()[0])

# ======================
# 4. FORMAT TANGGAL
# ======================
df['Tanggal_Ujian'] = pd.to_datetime(
    df['Tanggal_Ujian'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

# cek ulang
print(df.isnull().sum())
print(df['Tanggal_Ujian'].head())
print(df.info())

# ======================
# 5. ENCODING
# ======================
cols = ['Nama', 'Jenis_Kelamin', 'Prodi', 'Status', 'Nilai_Akhir']

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"Mapping {col}:",
          dict(zip(le.classes_, le.transform(le.classes_))))

# ======================
# 6. SPLIT DATA
# ======================
X = df.drop(['Nilai_Akhir', 'ID'], axis=1)
y = df['Nilai_Akhir']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ======================
# 7. VISUALISASI
# ======================
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.hist(df['Umur'])
plt.title('Distribusi Umur')
plt.hist(df['Umur'], bins=10)

plt.subplot(1, 2, 2)
df['Prodi'].value_counts().plot(kind='bar')
plt.title('Jumlah Mahasiswa per Prodi')
plt.xticks(
    ticks=range(4),
    labels=['Data Science', 'Informatika', 'Sistem Informasi', 'Teknik Komputer'],
    rotation=45
)

plt.tight_layout()
plt.show()