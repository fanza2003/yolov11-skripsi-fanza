import sqlite3
import hashlib
import PIL
import streamlit as st
from pathlib import Path
import json
import helper
import settings
from PIL import Image
from ultralytics import YOLO

# Layout
st.set_page_config(
    page_title="Object Detection menggunakan YOLOv11",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auth
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user[0] if user and user[1] == hash_password(password) else None

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
    st.session_state['username'] = None
    st.session_state['name'] = None

if st.session_state['authentication_status'] != True:
    st.header("Login 🍎")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        name = verify_user(username, password)
        if name:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.session_state['name'] = name
            st.success("Berhasil login")
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")

else:
    def show_mobile_warning():
        st.markdown("""
            <style>
            #mobile-warning {
                color: red;
                font-weight: bold;
                margin-top: 8px;
            }
            </style>

            <div id="mobile-warning" style="display:none;">
                ⚠️ Pengguna HP wajib menggunakan mode desktop agar aplikasi berfungsi dengan baik.
            </div>

            <script>
            function isMobile() {
                return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            }
            if(isMobile()){
                document.getElementById("mobile-warning").style.display = "block";
            }
            </script>
        """, unsafe_allow_html=True)

    def main():
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        st.sidebar.title(f"Welcome, {st.session_state['name']}")
        st.sidebar.header("🍎 Apel Indonesia")

        if st.sidebar.button("Logout"):
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.experimental_rerun()

        menu_options = {
            "Home": "🏠 Beranda",
            "Detection": "🔍 Object Detection",
            "History": "📜 Riwayat"
        }
        selected_menu = st.sidebar.radio("Select Menu", list(menu_options.keys()), format_func=lambda x: menu_options[x])

        if selected_menu == "Home":
            st.header("Selamat datang di aplikasi YOLOv11 untuk deteksi penyakit apel!")
            st.markdown("""
            Aplikasi ini mendeteksi penyakit apel menggunakan YOLOv11 mode utama:  
            **Object Detection** – Mendeteksi dan memberi bounding box pada penyakit pada buah apel.  
            
            Silakan pilih mode yang Anda inginkan melalui menu di sidebar.
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.image("images/yak apple.jpg", caption="Overview Image", use_column_width=True)
            with col2:
                st.image("images/yak apple hasil.png", caption="Overview Deteksi", use_column_width=True)

        elif selected_menu == "Detection":
            st.sidebar.header("ML Model Config")
            confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
            model_path = Path(settings.DETECTION_MODEL)

            try:
                model = helper.load_model(model_path)
            except Exception as ex:
                st.error(f"Unable to load model: {model_path}")
                st.error(ex)
                return

            st.sidebar.header("Input Method")
            input_method = st.sidebar.radio("Pilih metode input gambar:", ["Upload Gambar", "Kamera Langsung"])

            if input_method == "Upload Gambar":
                source_img = st.file_uploader("Pilih gambar..", type=("jpg", "jpeg", "png"))
                show_mobile_warning()
                
                if source_img:
                    img = PIL.Image.open(source_img)
                    if st.button("Detect Objects"):
                        res = model.predict(img, conf=confidence)
                        boxes = res[0].boxes
                        plotted = res[0].plot()[:, :, ::-1]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img, caption="Gambar yang diupload", use_column_width=True)
                        with col2:
                            st.image(plotted, caption="Hasil Deteksi", use_column_width=True)

                        try:
                            with open("penyakit_apple_info.json", "r", encoding="utf-8") as f:
                                penyakit_info = json.load(f)
                            detected_labels = set()
                            for box in boxes:
                                cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                                label = model.names[cls]
                                detected_labels.add(label)

                            penjelasan_list = []
                            for label in detected_labels:
                                if label in penyakit_info:
                                    penjelasan_list.append(f"**{label}**: {penyakit_info[label]}")
                                else:
                                    penjelasan_list.append(f"**{label}**: Info tidak tersedia")

                            if 'history' not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append({
                                "image": img,
                                "result": plotted,
                                "boxes": boxes,
                                "penjelasan": penjelasan_list
                            })

                            st.markdown("### 🧠 Penjelasan Penyakit Terdeteksi")
                            for p in penjelasan_list:
                                st.info(p)

                        except:
                            st.warning("File penyakit_apple_info.json tidak ditemukan")

            elif input_method == "Kamera Langsung":
                camera_image = st.camera_input("Ambil Foto dengan Kamera")
                if camera_image:
                    camera_img = PIL.Image.open(camera_image)
                    res_cam = model.predict(camera_img, conf=confidence)
                    boxes_cam = res_cam[0].boxes
                    plotted_cam = res_cam[0].plot()[:, :, ::-1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(camera_img, caption="Gambar dari Kamera", use_column_width=True)
                    with col2:
                        st.image(plotted_cam, caption="Hasil Deteksi dari Kamera", use_column_width=True)

                    try:
                        with open("penyakit_apple_info.json", "r", encoding="utf-8") as f:
                            penyakit_info = json.load(f)
                        detected_labels = set()
                        for box in boxes_cam:
                            cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                            label = model.names[cls]
                            detected_labels.add(label)

                        penjelasan_list = []
                        for label in detected_labels:
                            if label in penyakit_info:
                                penjelasan_list.append(f"**{label}**: {penyakit_info[label]}")
                            else:
                                penjelasan_list.append(f"**{label}**: Info tidak tersedia")

                        if 'history' not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "image": camera_img,
                            "result": plotted_cam,
                            "boxes": boxes_cam,
                            "penjelasan": penjelasan_list
                        })

                        st.markdown("### 🧠 Penjelasan Penyakit Terdeteksi")
                        for p in penjelasan_list:
                            st.info(p)

                    except:
                        st.warning("File penyakit_apple_info.json tidak ditemukan")

        elif selected_menu == "Segmentation":
            st.sidebar.header("ML Model Config")
            confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
            model_path = Path(settings.SEGMENTATION_MODEL)

            try:
                model = helper.load_model(model_path)
            except Exception as ex:
                st.error(f"Unable to load segmentation model: {model_path}")
                st.error(ex)
                return

            st.sidebar.header("Input Method")
            input_method = st.sidebar.radio("Pilih metode input gambar:", ["Upload Gambar", "Kamera Langsung"])

            if input_method == "Upload Gambar":
                source_img = st.file_uploader("Pilih gambar...", type=("jpg", "jpeg", "png", "bmp", "webp"))
                show_mobile_warning()

                if source_img:
                    img = PIL.Image.open(source_img)
                    if st.button("Segment Image"):
                        res = model.predict(img, conf=confidence)
                        plotted = res[0].plot()[:, :, ::-1]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img, caption="Gambar yang diupload", use_column_width=True)
                        with col2:
                            st.image(plotted, caption="Hasil Segmentasi", use_column_width=True)

                        if 'history' not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({"image": img, "result": plotted})

            elif input_method == "Kamera Langsung":
                camera_image = st.camera_input("Ambil Foto dengan Kamera")
                if camera_image:
                    camera_img = PIL.Image.open(camera_image)
                    res_cam = model.predict(camera_img, conf=confidence)
                    plotted_cam = res_cam[0].plot()[:, :, ::-1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(camera_img, caption="Gambar dari Kamera", use_column_width=True)
                    with col2:
                        st.image(plotted_cam, caption="Hasil Segmentasi dari Kamera", use_column_width=True)

                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({"image": camera_img, "result": plotted_cam})

        elif selected_menu == "History":
            st.header("Detection History")
            if st.session_state.get('history'):
                for idx, rec in enumerate(st.session_state.history):
                    st.subheader(f"Record {idx + 1}")
                    st.image(rec['image'], caption=f"Image {idx + 1}", use_column_width=True)
                    st.image(rec['result'], caption=f"Result {idx + 1}", use_column_width=True)
                    if 'boxes' in rec:
                        with st.expander(f"Boxes Detail {idx + 1}"):
                            for box in rec['boxes']:
                                st.write(box.data)
                    if 'penjelasan' in rec:
                        with st.expander(f"Penjelasan Penyakit {idx + 1}"):
                            for p in rec['penjelasan']:
                                st.markdown(p)
            else:
                st.write("Belum ada riwayat deteksi.")

        # Dark mode toggle
        st.sidebar.markdown("---")
        st.session_state.dark_mode = st.sidebar.checkbox('Dark Mode', value=st.session_state.dark_mode)
        if st.session_state.dark_mode:
            st.sidebar.markdown("""<p style="color:white; font-size:12px;">❗ Gunakan saat Streamlit dalam mode gelap ❗</p>""", unsafe_allow_html=True)
            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {background-color:#1E1E1E; color:#FFF;}
            [data-testid="stSidebar"] {background-color:#333; color:#FFF;}
            [data-testid="stExpander"] {background-color:#2E2E2E;}
            </style>""", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""<p style="color:black; font-size:12px;">❗ Gunakan saat Streamlit dalam mode terang ❗</p>""", unsafe_allow_html=True)
            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {background-color:#ffffe0; color:#000;}
            [data-testid="stSidebar"] {background-color:#FFA62F; color:#000;}
            </style>""", unsafe_allow_html=True)

        st.sidebar.image("images/poon.png", use_column_width=True)

    if __name__ == "__main__":
        main()
