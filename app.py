import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt


# Judul aplikasi
st.title("Klasifikasi Jenis Tanah Gambut")
st.markdown("Model CNN Untuk Klasifikasi Tanah Gambut: **Fibrik**, **Hemik**, dan **Saprik**.")

st.image("https://upload.wikimedia.org/wikipedia/commons/3/35/Tanah_gambut.jpg", use_container_width=True)

# Sidebar Menu
menu = st.sidebar.selectbox("Navigasi", ["Beranda", "Upload Gambar", "Grafik Model", "Tentang"])

# Fungsi Prediksi
def prediksi_tanah(image, model):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    kelas = ['Fibrik', 'Hemik', 'Saprik']
    hasil = kelas[np.argmax(prediction)]
    probabilitas = prediction[0][np.argmax(prediction)]
    return hasil, probabilitas, prediction[0]

# Beranda
if menu == "Beranda":
    if os.path.exists("gambar_logo.png"):
        st.image("gambar_logo.png", use_container_width=True)
    else:
        st.warning("Gambar logo tidak ditemukan.")
    st.markdown("**Aplikasi ini digunakan untuk mengklasifikasikan jenis tanah gambut** berdasarkan gambar menggunakan Convolutional Neural Network (CNN).")

# Upload Gambar
elif menu == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar tanah (jpg/png)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        if st.button("Prediksi"):
            try:
                model = tf.keras.models.load_model("cnn_tanah_gambut.h5")
                hasil, prob, all_probs = prediksi_tanah(image, model)

                st.success(f"Jenis Tanah: **{hasil}**")
                st.info(f"Probabilitas: {prob * 100:.2f}%")

                # Visualisasi Probabilitas
                st.subheader("Probabilitas Tiap Kelas")
                fig, ax = plt.subplots()
                kelas = ['Fibrik', 'Hemik', 'Saprik']
                ax.bar(kelas, all_probs, color='skyblue')
                ax.set_ylabel("Probabilitas")
                st.pyplot(fig)

                # Tombol unduh hasil
                hasil_text = f"Jenis Tanah: {hasil}\nProbabilitas: {prob*100:.2f}%"
                st.download_button("Unduh Hasil Prediksi", hasil_text, file_name="hasil_prediksi.txt")

            except Exception as e:
                st.error(f"Gagal memuat model atau memproses gambar: {e}")

# Grafik Model
elif menu == "Grafik Model":
    st.subheader("Grafik Akurasi dan Loss Model")

    try:
        with open("riwayat_pelatihan.pkl", "rb") as file:
            history = pickle.load(file)

        # Grafik Akurasi
        fig1, ax1 = plt.subplots()
        ax1.plot(history['accuracy'], label='Akurasi Training')
        ax1.plot(history['val_accuracy'], label='Akurasi Validasi')
        ax1.set_title("Akurasi")
        ax1.legend()
        st.pyplot(fig1)

        # Grafik Loss
        fig2, ax2 = plt.subplots()
        ax2.plot(history['loss'], label='Loss Training')
        ax2.plot(history['val_loss'], label='Loss Validasi')
        ax2.set_title("Loss")
        ax2.legend()
        st.pyplot(fig2)

    except FileNotFoundError:
        st.warning("File riwayat_pelatihan.pkl tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menampilkan grafik: {e}")

# Tentang
elif menu == "Tentang":
    st.subheader("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan oleh **Ismi Asmita** dari Universitas Pasir Pengaraian. 
    Tujuan utama aplikasi ini adalah untuk membantu klasifikasi jenis tanah gambut seperti:
    - Fibrik
    - Hemik
    - Saprik

    **Teknologi yang digunakan:**
    - Convolutional Neural Network (CNN)
    - Streamlit untuk antarmuka pengguna
    - TensorFlow & Keras untuk pelatihan model
    - Matplotlib & Pickle untuk visualisasi

    **Kontak & Sumber:**
    - GitHub: [Ismi100603](https://github.com/Ismi100603)
    """)
