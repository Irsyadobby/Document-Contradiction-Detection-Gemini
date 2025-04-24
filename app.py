import streamlit as st
import pandas as pd
import numpy as np
import os
import io
# import glob
# import re

import google.generativeai as gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.api_core.exceptions
import time

# --- Konfigurasi ---

# !!! PERINGATAN KEAMANAN !!!
# Menggunakan API Key secara hardcoded seperti di bawah ini SANGAT BERISIKO.
# Jangan pernah melakukan ini pada kode yang akan dibagikan, diunggah ke Git,
# atau digunakan dalam lingkungan produksi/deployment.
# Kunci Anda dapat dengan mudah terekspos dan disalahgunakan.
# Ganti string di bawah dengan API Key Anda yang sebenarnya.
API_KEY = "AIzaSyCqBYY_7oyE0rUiI-79U9jvnPCxXjHKAIE"

# Menampilkan peringatan di aplikasi Streamlit tentang risiko keamanan
# st.error("PERINGATAN: API Key di-hardcode langsung dalam skrip! Ini sangat tidak aman dan hanya boleh digunakan untuk pengujian pribadi yang sangat terbatas.", icon="🚨")

# Validasi sederhana untuk memastikan placeholder sudah diganti
if not API_KEY or API_KEY == "GANTI_DENGAN_API_KEY_ANDA_YANG_SEBENARNYA_DISINI":
    st.error("API Key belum dimasukkan! Harap ganti placeholder 'GANTI_DENGAN_API_KEY_ANDA_YANG_SEBENARNYA_DISINI' di dalam kode dengan API Key Anda.", icon="🛑")
    st.stop() # Hentikan aplikasi jika API Key belum diisi

# Langsung konfigurasikan Gemini dengan API Key yang di-hardcode
try:
    gemini.configure(api_key=API_KEY)
    # Anda bisa menambahkan st.info jika ingin konfirmasi, tapi mungkin tidak perlu
    # st.info("Gemini dikonfigurasi menggunakan API Key hardcoded.", icon="🔧")
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Gemini dengan API Key yang diberikan: {e}", icon="🛑")
    st.stop() # Hentikan aplikasi jika konfigurasi gagal

# --- Sisa Kode Anda ---
# (Nama Model, Konfigurasi Generasi/Keamanan, Inisialisasi Model,
#  Fungsi generate_relationship, handle_text_input,
#  analyze_clusters_streamlit, DataFrameInput, main, dll.
#  TETAP SAMA seperti kode sebelumnya)

MODEL_NAME = "gemini-1.5-flash-latest"
# MODEL_NAME = "gemini-pro" # Alternatif

generation_config = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# Inisialisasi Model Gemini (setelah konfigurasi)
try:
    model = gemini.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception as e:
    st.error(f"Gagal menginisialisasi model Gemini '{MODEL_NAME}': {e}", icon="🛑")
    st.stop()


# --- Fungsi Inti Gemini AI (Diperbaiki) ---
def generate_relationship(content1, content2):
    """
    Menentukan hubungan antara dua teks menggunakan model Gemini.

    Args:
        content1 (str): Teks pertama.
        content2 (str): Teks kedua.

    Returns:
        tuple: (sentiment, reason) atau (None, None) jika terjadi error,
               format tidak sesuai, atau input tidak valid.
               sentiment bisa berupa 'Kontradiktif', 'Pro', 'Netral'.
    """
    if not isinstance(content1, str) or not isinstance(content2, str) or not content1 or not content2:
        st.warning("Input teks tidak valid atau kosong.", icon="⚠️")
        return None, None

    # Prompt yang diperjelas dan memasukkan teks langsung
    prompt = f"""Analisis hubungan antara dua teks berikut. Tentukan apakah keduanya:
1.  **Kontradiktif**: Saling bertentangan secara langsung atau menyajikan informasi yang tidak konsisten.
2.  **Pro**: Saling mendukung, menguatkan, atau menyatakan hal yang sejalan.
3.  **Netral**: Tidak berhubungan secara signifikan, membahas topik yang sama sekali berbeda, atau hubungan tidak jelas.

Berikan jawaban HANYA dalam format: **LABEL|Alasan singkat dan jelas.**

Contoh Format Jawaban:
Kontradiktif|Teks pertama menyatakan X terjadi, sedangkan teks kedua menyatakan X tidak terjadi.
Pro|Kedua teks membahas dampak positif dari Y pada sektor Z.
Netral|Teks pertama tentang regulasi impor, teks kedua tentang festival budaya.

---
Teks 1:
"{content1}"
---
Teks 2:
"{content2}"
---

Hubungan (LABEL|Alasan): """

    try:
        # Panggil API
        responses = model.generate_content(prompt)

        # Ekstraksi teks respons dengan aman
        response_text = ""
        if responses.parts:
             response_text = responses.text.strip()
        # Fallback jika tidak ada 'parts' tapi ada 'text' (jarang terjadi)
        elif hasattr(responses, 'text') and responses.text:
             response_text = responses.text.strip()
        # Tangani kasus respons diblokir atau kosong
        else:
            reason_blocked = "Tidak diketahui"
            if responses.prompt_feedback and responses.prompt_feedback.block_reason:
                reason_blocked = responses.prompt_feedback.block_reason.name
            st.warning(f"Respons dari API kosong atau diblokir. Alasan: {reason_blocked}", icon="🚫")
            # Anda bisa mencetak detail lebih lanjut jika perlu:
            # st.write("Detail Respons:", responses)
            return None, f"Respons API kosong/diblokir ({reason_blocked})"

        # Parsing respons dengan pemisah '|'
        if '|' in response_text:
            parts = response_text.split('|', 1)
            sentiment = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else "Alasan tidak diberikan oleh model."

            # Validasi label sentiment
            valid_sentiments = ["Kontradiktif", "Pro", "Netral"]
            if sentiment not in valid_sentiments:
                 st.warning(f"Label sentimen tidak dikenal ('{sentiment}') dalam respons: '{response_text}'. Dianggap Netral.", icon="❓")
                 # Bisa dikembalikan sebagai Netral atau None, sesuai kebutuhan
                 return "Netral", f"Label tidak valid: {response_text}"
            return sentiment, reason
        else:
            # Jika format tidak sesuai, anggap Netral dan sertakan respons asli
            st.warning(f"Format respons tidak sesuai: '{response_text}'. Dianggap Netral.", icon="❓")
            return "Netral", f"Format tidak sesuai: {response_text}"

    # Tangani error spesifik dari API Google
    except google.api_core.exceptions.GoogleAPICallError as e:
        st.error(f"Error API Google: {e}", icon="🔥")
        return None, f"Error API: {e}"
    # Tangani error umum lainnya
    except Exception as e:
        st.error(f"Terjadi error tak terduga saat memanggil Gemini API: {e}", icon="💥")
        # import traceback
        # st.exception(traceback.format_exc()) # Tampilkan traceback untuk debug
        return None, f"Error: {e}"

# --- Fungsi untuk Tab Input Teks (Diperbaiki) ---
def handle_text_input(content1, content2):
    """Memproses input teks tunggal dan menampilkan hasilnya."""
    with st.spinner("Menganalisis hubungan teks..."):
        sentiment, reason = generate_relationship(content1, content2)

    if sentiment is not None:
        # Tampilkan hasil dengan format yang lebih jelas
        st.subheader("Hasil Analisis:")
        col1, col2 = st.columns([1, 3]) # Beri lebih banyak ruang untuk alasan
        with col1:
            if sentiment == "Pro":
                st.success(f"**Hubungan:**\n{sentiment}", icon="✅")
            elif sentiment == "Kontradiktif":
                st.error(f"**Hubungan:**\n{sentiment}", icon="❌")
            else: # Netral atau lainnya
                st.info(f"**Hubungan:**\n{sentiment}", icon="⚪")
        with col2:
            st.write("**Alasan:**")
            st.markdown(f"> {reason}") # Gunakan markdown untuk blockquote
    else:
        # Tampilkan pesan error jika analisis gagal
        st.error(f"Gagal menganalisis teks. Alasan: {reason}", icon="⚠️")


# --- Fungsi untuk Tab Upload File (Diperbaiki) ---
def analyze_clusters_streamlit(df):
    """Menganalisis hubungan teks dalam setiap cluster pada DataFrame."""
    summary = {"Kontradiktif": 0, "Pro": 0, "Netral": 0, "Error/Skip": 0}
    results_detail = [] # Untuk menyimpan detail hasil per pasangan

    if "Cluster" not in df.columns or "Teks" not in df.columns:
        st.error("Error: DataFrame harus memiliki kolom 'Cluster' dan 'Teks'.", icon="🛑")
        return summary, pd.DataFrame(results_detail) # Kembalikan summary dan df kosong

    cluster_groups = df.groupby("Cluster")
    total_pairs_to_process = sum(len(group)*(len(group)-1)//2 for _, group in cluster_groups)
    processed_pairs = 0

    st.info(f"Total pasangan teks yang akan dianalisis: {total_pairs_to_process}")
    progress_bar = st.progress(0.0) # Inisialisasi progress bar
    status_text = st.empty() # Placeholder untuk teks status

    start_time = time.time()

    for cluster, group in cluster_groups:
        texts = group["Teks"].dropna().astype(str).tolist() # Pastikan string & hilangkan NaN
        n_texts = len(texts)
        status_text.text(f"Memproses Cluster {cluster} ({n_texts} teks)...")

        if n_texts < 2:
            st.write(f"Cluster {cluster}: Kurang dari 2 teks, dilewati.")
            continue

        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                processed_pairs += 1
                status_text.text(f"Cluster {cluster}: Membandingkan teks {i+1} vs {j+1} ({processed_pairs}/{total_pairs_to_process})...")

                # Panggil fungsi generate_relationship
                sentiment, reason = generate_relationship(texts[i], texts[j])

                # Catat hasil detail
                results_detail.append({
                    "Cluster": cluster,
                    "Teks 1 Index": i,
                    "Teks 2 Index": j,
                    "Teks 1 Snippet": texts[i][:100] + "...", # Tampilkan potongan teks
                    "Teks 2 Snippet": texts[j][:100] + "...",
                    "Hasil Sentimen": sentiment if sentiment is not None else "Error",
                    "Alasan/Error": reason if reason is not None else "Tidak ada detail"
                })

                if sentiment is not None:
                    if sentiment in summary:
                        summary[sentiment] += 1
                    else:
                        # Jika ada label aneh (seharusnya sudah ditangani di generate_relationship)
                        st.warning(f"Cluster {cluster}, Teks {i+1} vs {j+1}: Sentimen tidak dikenal '{sentiment}'. Dicatat sebagai Error/Skip.")
                        summary["Error/Skip"] += 1
                else:
                    # Jika generate_relationship mengembalikan None (error)
                    summary["Error/Skip"] += 1

                # Update progress bar
                progress_bar.progress(processed_pairs / total_pairs_to_process if total_pairs_to_process > 0 else 0)

                # Opsional: Tambahkan jeda kecil untuk menghindari rate limit API
                # time.sleep(0.5) # Jeda 0.5 detik

    end_time = time.time()
    elapsed_time = end_time - start_time
    status_text.text(f"Analisis selesai dalam {elapsed_time:.2f} detik.")
    progress_bar.progress(1.0) # Pastikan progress bar penuh

    return summary, pd.DataFrame(results_detail)


def DataFrameInput():
    """Menangani UI untuk upload file dan analisis cluster."""
    # Pastikan openpyxl terinstal
    try:
        import openpyxl
    except ImportError:
        st.error("Library 'openpyxl' diperlukan untuk fitur download Excel. Silakan install dengan `pip install openpyxl` dan restart aplikasi.", icon="⚠️")
        st.stop()

    data_up = st.file_uploader("Upload File Teks (CSV atau XLSX)", type=['csv', 'xlsx'], accept_multiple_files=False, key="file_uploader")

    if data_up:
        try:
            file_extension = os.path.splitext(data_up.name)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(data_up)
            elif file_extension == '.xlsx':
                df = pd.read_excel(data_up, engine='openpyxl')
            else:
                st.error("Format file tidak didukung.", icon="❌")
                return

            if not {'Teks', 'Cluster'}.issubset(df.columns):
                st.error("File harus memiliki kolom 'Teks' dan 'Cluster'.", icon="⚠️")
                return

            st.write("Pratinjau Data:")
            st.dataframe(df.head())

            if st.button("Mulai Analisis File", key="analyze_button"):
                summary, results_df = analyze_clusters_streamlit(df.copy()) # Gunakan copy

                st.subheader("Ringkasan Hubungan Antar Teks per Cluster")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Kontradiktif", summary["Kontradiktif"])
                col2.metric("Pro", summary["Pro"])
                col3.metric("Netral", summary["Netral"])
                col4.metric("Error/Skip", summary["Error/Skip"])

                st.subheader("Detail Hasil Analisis per Pasangan Teks")
                st.dataframe(results_df)

                # --- Tombol Download ---
                st.write("") # Beri sedikit spasi

                if not results_df.empty: # Hanya tampilkan tombol jika ada hasil
                    # Fungsi untuk konversi ke byte Excel (cached)
                    @st.cache_data
                    def convert_df_to_excel(df_to_convert):
                        output = io.BytesIO()
                        # Gunakan ExcelWriter untuk kontrol lebih (opsional, bisa langsung to_excel)
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_to_convert.to_excel(writer, index=False, sheet_name='Hasil Analisis')
                        # Ambil nilai byte setelah selesai menulis
                        processed_data = output.getvalue()
                        return processed_data

                    # Fungsi konversi ke CSV (tetap ada)
                    @st.cache_data
                    def convert_df_to_csv(df_to_convert):
                        return df_to_convert.to_csv(index=False).encode('utf-8')

                    # Siapkan data byte untuk diunduh
                    excel_data_bytes = convert_df_to_excel(results_df)
                    csv_data_bytes = convert_df_to_csv(results_df)

                    # Tampilkan tombol download dalam kolom agar rapi
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        st.download_button(
                            label="📥 Unduh Hasil (Excel)",
                            data=excel_data_bytes,
                            file_name='hasil_analisis_hubungan_teks.xlsx',
                            # MIME type untuk file .xlsx
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key='download_excel_button' # Beri key unik
                        )

                    with col_dl2:
                        st.download_button(
                            label="📥 Unduh Hasil (CSV)",
                            data=csv_data_bytes,
                            file_name='hasil_analisis_hubungan_teks.csv',
                            mime='text/csv',
                            key='download_csv_button' # Beri key unik
                        )
                else:
                    st.info("Tidak ada hasil detail untuk diunduh.")


        except Exception as e:
            st.error(f"Terjadi error saat memproses file: {e}", icon="💥")
            # st.exception(e) # Uncomment untuk melihat traceback lengkap saat debug

# --- App Front End (Struktur Utama) ---
def main():
    st.set_page_config(page_title="Analisis Hubungan Teks", layout="wide", initial_sidebar_state="collapsed")

    st.title("🔬 Analisis Hubungan Antar Teks (Natural Language Inference)")
    st.markdown("""
    Aplikasi ini menggunakan Model AI Generatif (Gemini) untuk menentukan hubungan
    antara dua teks (Kontradiktif, Pro, atau Netral). Anda bisa memasukkan teks
    secara manual atau mengunggah file CSV/XLSX yang berisi kolom 'Teks' dan 'Cluster'.
    """)

    # Peringatan tentang API Key jika masih hardcoded
    API_KEY = 'AIzaSyCqBYY_7oyE0rUiI-79U9jvnPCxXjHKAIE' # Sesuaikan jika Anda mengubah nilai hardcoded
         # st.error("PERINGATAN KERAS: Anda menggunakan API Key yang di-hardcode. Ini sangat tidak aman untuk produksi atau jika kode dibagikan. Harap konfigurasikan Streamlit Secrets atau Environment Variable.", icon="🚨")

    tab1, tab2 = st.tabs(['Analisis Dua Teks', 'Analisis File (Cluster)'])

    # Tab 1: Input Teks Manual
    with tab1:
        st.header("Masukkan Dua Teks untuk Dianalisis")
        col1, col2 = st.columns(2)
        with col1:
            text1 = st.text_area("Teks 1", placeholder="Masukkan teks pertama di sini...", height=200, key="text1_input")
        with col2:
            text2 = st.text_area("Teks 2", placeholder="Masukkan teks kedua di sini...", height=200, key="text2_input")

        if st.button("Analisis Teks Manual", key="submit_text_button"):
            if text1 and text2:
                handle_text_input(text1, text2)
            else:
                st.warning("Harap masukkan kedua teks sebelum menganalisis.", icon="⚠️")

    # Tab 2: Upload File
    with tab2:
        st.header("Unggah File untuk Analisis per Cluster")
        st.markdown("""
        Unggah file `.csv` atau `.xlsx` yang memiliki kolom **'Teks'** (berisi teks yang akan dianalisis)
        dan **'Cluster'** (berisi label cluster untuk setiap teks). Aplikasi akan membandingkan
        setiap pasangan teks dalam cluster yang sama.
        """)
        DataFrameInput() # Panggil fungsi yang menangani UI upload file

    st.markdown("---")
    st.caption("Dibuat menggunakan Streamlit dan Google Gemini")


if __name__ == "__main__":
    main()