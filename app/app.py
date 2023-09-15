# Import streamlit
import streamlit as st
from streamlit_option_menu import option_menu
from fastai.vision.all import *
from fastai.learners import load_learner
from cats_and_dogs_breed import upload_photo, capture_photo, model_info


def app():
    #######################################################################################################################

    # Set the page config
    st.set_page_config(
        page_title="Cat & Dog Breed Detector",  # The title of the web page
        page_icon="🐶",  # The icon of the web page, can be an emoji or a file path
        initial_sidebar_state="collapsed"
    )

    #######################################################################################################################

    st.markdown("<h1 style='text-align: center;'>🐶🐱 Cat & Dog Breed Detector 🐱🐶</h1>",
                unsafe_allow_html=True)

    #######################################################################################################################

    # Options Menu at the top of the homepage
    selected = option_menu(None, ["Upload", "Capture", "Model"],
                           icons=["cloud upload", "camera", "gear"],
                           menu_icon="cast", default_index=0, orientation="horizontal")

    #######################################################################################################################

    # Load model and model class labels (vocab)
    model = load_learner(fname="models/pets_breed_learner.pkl")

    with open("models/pets_breed_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    # Sorting
    vocab = sorted(vocab)

    #######################################################################################################################

    if selected == "Upload":
        st.caption("""**Experience the power of our Cat & Dog Breed Detector**, a cutting-edge deep learning model based on
                   the **FastAi framework** and the **ResNet-50** architecture. This advanced model showcases **remarkable performance**,
                   achieving an accuracy of over 93% while classifying pet images into **37 unique categories** representing various
                   cat and dog breeds. Trained on the **challenging Oxford-IIIT Pet Dataset** comprising a substantial **7390
                   high-quality pet images**, our project offers a fascinating glimpse into the world of pet breeds.
                   Whether you're a pet enthusiast or simply curious about your furry friends, our Cat & Dog Breed Detector
                   provides an engaging and informative experience for all. Discover your pet's true breed with confidence
                   using our state-of-the-art model!""")

        upload_photo(model=model, vocab=vocab, key="upload photo")

    #######################################################################################################################

    if selected == "Capture":
        capture_photo(model=model, vocab=None, key="capture photo")

    if selected == "Model":
        model_info()


#######################################################################################################################
if __name__ == "__main__":
    app()
