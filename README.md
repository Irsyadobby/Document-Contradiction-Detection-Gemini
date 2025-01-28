# Document-Contradiction-Detection-IndoBERT-LDA

Analisis menyeluruh terhadap keselarasan antar peraturan-peraturan dengan norma dasar dan hierarki perundang-undangan penting untuk memastikan implementasi perubahan undang-undang berjalan efisien dan konsisten dengan sistem hukum yang berlaku.

Pada penelitian ini akan digunakan machine learning dalam analisis keselarasan perundang-undangan untuk memfasilitasi identifikasi potensi disharmoni atau ketidakselarasan antar undang-undang bidang investasi dengan lebih efisien dan efektif. Metode yang akan digunakan untuk mengidentifikasi potensi disharmoni atau ketidakselarasan antar undang-undang adalah Latent Dirichlet Allocation (LDA) untuk clustering topik undang-undang dan Model Natural Language Inference (NLI) IndoBERT untuk menilai hubungan antar ayat.


## Metodologi
![Image](https://github.com/user-attachments/assets/0f1f429f-bc05-4d30-991c-4c98290e78ba)

Flowchart menggambarkan alur analisis data yang digunakan untuk mendeteksi potensi disharmoni pada peraturan perundang-undangan di bidang investasi. Proses dimulai dengan mengumpulkan dan mengekstraksi dokumen peraturan perundang-undangan yang digunakan pada penelitian ini. Kemudian dilakukan eksplorasi untuk melihat pola dan gambaran umum dari dokumen perundang-undangan yang digunakan. Selanjutnya, dokumen-dokumen ini akan melalui tahap pre-processing untuk membersihkan teks dan mengubahnya menjadi format yang lebih sesuai untuk analisis.

Setelah proses pre-processing, data frame berisi teks perundang-undangan tersebut akan digunakan untuk melatih model Latent Dirichlet Allocation (LDA). Model LDA akan mengidentifikasi topik-topik utama yang muncul dalam korpus dokumen perundang-undangan, sehingga dapat membagi dokumen menjadi kelompok-kelompok berdasarkan konteks yang dibahasnya. Kemudian akan dilakukan tahap filtering kelompok atau cluster dengan menghitung jarak Euclidean dari nilai probabilitas tiap dua objek (ayat) dan menetapkan nilai cut off agar hanya pasangan ayat dengan nilai jarak yang minimum (tingkat kemiripan tinggi) yang digunakan pada tahapan Fine Tuning. Selanjutnya, akan dilakukan tahap analisis menggunakan Model Natural Language Inference (NLI) IndoBERT. Model NLI ini akan membantu dalam mengevaluasi implikasi dan potensi kontradiksi antara ayat-ayat dalam dokumen perundang-undangan
