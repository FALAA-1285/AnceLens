# AcneLens

Deteksi dan klasifikasi **jerawat papula, pustula, dan kulit normal** menggunakan
**SURF + HOG + SVM**. Project ini mengimplementasikan ulang jurnal:

> A. Luthfi & M. Ali, "Analisis Kinerja Ekstraksi Fitur Speeded Up Robust Features (SURF) dan
> Histogram of Oriented Gradients (HOG) pada Deteksi Acne Vulgaris", *Jurnal Andalas: Rekayasa dan
> Penerapan Teknologi*, Vol. 06 No. 01, 2026. (`docs/145 Prod-1.pdf`)

## Struktur folder

```
AcneLens/
├── data/
│   ├── raw/                  # dataset asli dari Kaggle (otomatis diunduh oleh notebook)
│   └── processed/            # 1.500 citra terpilih (non-acne/papula/pustula), split.csv, fitur.joblib
├── docs/                     # jurnal acuan
├── models/                   # model SVM terbaik (model_terbaik.joblib)
├── notebooks/
│   └── acne_detection.ipynb  # seluruh tahapan: download dataset -> evaluasi
├── reports/
│   ├── figures/              # grafik akurasi & confusion matrix
│   ├── tabel3_similarity.csv
│   └── tabel4_akurasi.csv
├── requirements.txt
└── README.md
```

## Metode (sesuai jurnal)

| Tahap | Detail |
|---|---|
| Dataset | 500 Non-acne (Skin Disease Detection Dataset), 500 Papula & 500 Pustula (Acne_dataset); 80% latih / 20% uji |
| Preprocessing | Grayscale → interpolasi resize 256×300 (bicubic / bilinear / Lanczos3) → CLAHE (transformasi uji) |
| SURF | Keypoint det(H) ≥ 1000 → keypoint terkuat → scale → ROI → 64 fitur |
| HOG | ROI 64×64, 9 bin, blok 2×2, cellsize 4/8/16/32 |
| Klasifikasi | Fitur SURF + HOG dinormalisasi → SVM kernel linear / polynomial / RBF, 5-fold cross-validation |
| Evaluasi | Akurasi latih & uji (Tabel 4), confusion matrix, similarity fitur asli vs CLAHE (Tabel 3) |

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook notebooks/acne_detection.ipynb
```

Lalu jalankan semua cell (*Run All*). Unduhan dataset ±2,8 GB. Jika Kaggle meminta login, buat API token
di kaggle.com → Settings → *Create New Token*, simpan ke `C:\Users\<nama>\.kaggle\kaggle.json`.

## Catatan implementasi

- **SURF ditulis sendiri** di notebook. OpenCV versi pip tidak menyertakan SURF karena termasuk algoritma *nonfree*.
- Lanczos3 memakai `Image.LANCZOS` dari Pillow (kernel Lanczos a = 3).
- Aturan keputusan mengikuti diagram alir pengujian: citra tanpa keypoint det(H) ≥ 1000 → "wajah tidak berjerawat".
