# Import streamlit
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
from fastai.vision.all import *
from fastai.learner import load_learner
import pickle

#######################################################################################################################


def upload_photo(model=None, vocab=None, key=None):
    options = st.multiselect("**Select multiple names at once, then do Google/Bing Search, Download, Upload & Detect ...**",
                             vocab,
                             vocab[10:18],
                             key="pets_multiselect")
    st.text(f"{options}")

    # Upload image
    uploaded_image = st.file_uploader(
        "**Upload aan image**", type=["jpg", "png", "jpeg"], key="pets_uploaded_image")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        if st.button("**Detect**", key="pets_detect"):
            label, index, preds = model.predict(image)

            values, indices = preds.topk(5)  # get the top 5 values and indices
            # get the corresponding class names
            top_classes = model.dls.vocab[indices]
            values = values.tolist()
            st.markdown(f"""<div style="text-align:center;">
                            <h1>{label}</h1>
                            </div>""",
                        unsafe_allow_html=True)

            st.image(
                image, caption=f'{label} {max(preds).item() * 100:.2f}%', use_column_width=True)
            st.write(top_classes)
            st.write(values)

#######################################################################################################################


def capture_photo(model=None, vocab=None, key=None):
    capture_toggle = st.toggle(
        label="Take a picture (`try to keep the subject at the center`)", key="pets_capture_photo")

    if capture_toggle:
        # Check if the cancel checkbox is not selected
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer:
            # To read image file buffer as a PIL Image:
            image = Image.open(img_file_buffer)

            st.image(image, use_column_width=True)

            if st.button(label="Detect", key="pets_capture_detect"):

                output = model.predict(image)
                st.markdown(f"""<div style="text-align:center;">
                                <h1>{output[0]}</h1>
                                </div>""",
                            unsafe_allow_html=True)

                st.image(
                    image, caption=f'{output[0]} {max(output[2]).item() * 100:.2f}%', use_column_width=True)


#######################################################################################################################


def model_info():
    # Model performance on Freezed Layers
    st.subheader("Model performance with Renset50 (freezed layers)")
    freezed_data = {
        'epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'train_loss': [0.666741, 0.496385, 0.483526, 0.375414, 0.289794, 0.215737, 0.156200, 0.110415, 0.078930, 0.065367],
        'valid_loss': [0.321287, 0.430292, 0.561120, 0.347090, 0.372382, 0.319737, 0.319586, 0.235808, 0.260270, 0.257863],
        'accuracy': [0.895129, 0.875507, 0.868065, 0.908660, 0.899188, 0.920839, 0.924899, 0.936401, 0.929635, 0.934371],
        'error_rate': [0.104871, 0.124493, 0.131935, 0.091340, 0.100812, 0.079161, 0.075101, 0.063599, 0.070365, 0.065629],
        'time': ['01:27', '01:26', '01:27', '01:28', '01:27', '01:25', '01:27', '01:27', '01:27', '01:26']
    }

    df = pd.DataFrame(freezed_data)
    st.table(df)
    st.line_chart(data=df, x="epoch", y=['train_loss', 'valid_loss', 'accuracy', 'error_rate'], height=600)


    # Model performance on Unfreezed Layers
    st.subheader("Model performance with Renset50 (unfreezed layers)")
    unfreezed_data = {
        'epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'train_loss': [0.057250, 0.052452, 0.043833, 0.050001, 0.044332, 0.043266, 0.043428, 0.032019, 0.035151, 0.036221],
        'valid_loss': [0.259445, 0.261673, 0.252830, 0.279817, 0.257765, 0.263906, 0.253806, 0.250571, 0.254164, 0.245009],
        'accuracy': [0.929635, 0.934371, 0.935047, 0.933694, 0.932341, 0.937077, 0.934371, 0.937754, 0.936401, 0.935047],
        'error_rate': [0.070365, 0.065629, 0.064953, 0.066306, 0.067659, 0.062923, 0.065629, 0.062246, 0.063599, 0.064953],
        'time': ['01:26', '01:26', '01:25', '01:26', '01:26', '01:27', '01:28', '01:29', '01:14', '01:03']
    }

    df = pd.DataFrame(unfreezed_data)
    st.table(df)
    st.line_chart(data=df, x="epoch", y=['train_loss', 'valid_loss', 'accuracy', 'error_rate'], height=600)


#######################################################################################################################
