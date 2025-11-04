EMNIST Letters Classification using HOG + SVM + LOOCV

Program ini bertujuan untuk mengklasifikasikan huruf tulisan tangan (A–Z) dari dataset EMNIST Letters menggunakan metode Histogram of Oriented Gradients (HOG) sebagai ekstraksi fitur, dan Support Vector Machine (SVM) sebagai model klasifikasi. Evaluasi dilakukan menggunakan Leave-One-Out Cross Validation (LOOCV) untuk mendapatkan hasil yang akurat dan menyeluruh.

Dataset

Dataset yang digunakan berasal dari EMNIST (Extended MNIST)
File yang digunakan dalam proyek ini:

1. emnist-letters-train.csv
Data utama berisi label dan nilai piksel 28×28.

2. emnist-letters-mapping.txt
Pemetaan label angka (1–26) ke huruf (A–Z).

Metodologi

1. Load Dataset → Membaca file CSV dan memisahkan label serta piksel gambar.

2. Perbaikan Orientasi → Rotasi dan flip gambar agar huruf tidak terbalik.

3. Sampling Seimbang → Mengambil jumlah data yang sama per kelas untuk menjaga keseimbangan.

4. Ekstraksi Fitur (HOG) → Mengubah gambar 28×28 menjadi vektor fitur berbasis arah gradien.

5. Standarisasi Data → Normalisasi fitur menggunakan StandardScaler.

6. Training SVM + Grid Search → Melatih model dan mencari parameter terbaik (C, kernel, gamma).

7. Evaluasi LOOCV → Menggunakan Leave-One-Out Cross Validation untuk menilai performa model.

8. Visualisasi → Menampilkan Confusion Matrix dan hasil metrik evaluasi (precision, recall, f1-score).

9. Model Saving → Menyimpan model dan scaler dengan joblib untuk digunakan kembali.


Hasil

1. Akurasi Total: 87.42%

2. Rata-rata Precision/Recall/F1-score: ±0.87

3. Model mampu mengenali huruf dengan baik, meskipun masih ada kesalahan pada huruf yang bentuknya mirip.

4. Evaluasi menggunakan LOOCV memastikan hasil yang lebih akurat dan tidak bias.