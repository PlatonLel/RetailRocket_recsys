import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict
from model import BPRMatrixFactorization 

MODEL_CHECKPOINT_PATH = "models/recommender_model_best.pt"
INFERENCE_DATA_DIR = "models/"
DEFAULT_K = 10

@st.cache_resource
def load_inference_data():
    try:
        with open(os.path.join(INFERENCE_DATA_DIR, "user_encoder.pkl"), "rb") as f:
            user_encoder = pickle.load(f)
        with open(os.path.join(INFERENCE_DATA_DIR, "item_encoder.pkl"), "rb") as f:
            item_encoder = pickle.load(f)
        with open(os.path.join(INFERENCE_DATA_DIR, "model_params.pkl"), "rb") as f:
            model_params = pickle.load(f)
        idx_to_item_id = {idx: item_id for idx, item_id in enumerate(item_encoder.classes_)}
        return user_encoder, item_encoder, idx_to_item_id, model_params
    except FileNotFoundError:
        st.error(f"Error: Could not find inference data in '{INFERENCE_DATA_DIR}'. Please ensure 'user_encoder.pkl', 'item_encoder.pkl', and 'model_params.pkl' exist.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading inference data: {e}")
        return None, None, None, None

@st.cache_resource
def load_model(model_params, checkpoint_path):
    if model_params is None:
        return None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BPRMatrixFactorization(
            num_users=model_params['num_users'],
            num_items=model_params['num_items'],
            embedding_dim=model_params['embedding_dim'],
            dropout_rate=model_params['dropout_rate']
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_recommendations(user_id_orig, model, device, k, user_encoder, item_encoder, idx_to_item_id):
    if model is None:
        return None, "Model not loaded."
    try:
        user_idx = user_encoder.transform([user_id_orig])[0]
    except ValueError:
        return None, f"User ID '{user_id_orig}' not found in training data."
    num_items = model.item_embeddings.num_embeddings
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
        all_item_indices = torch.arange(num_items, dtype=torch.long).to(device)
        user_rep = user_tensor.repeat(num_items)
        scores = model(user_rep, all_item_indices)
        top_scores, top_indices = torch.topk(scores, k)
        recommended_item_indices = top_indices.cpu().numpy()
        recommended_item_ids = [idx_to_item_id.get(idx, f"Unknown_Idx_{idx}") for idx in recommended_item_indices]
        return recommended_item_ids, None

st.set_page_config(page_title="Recommender System", page_icon="🛍️", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stTable {background-color: #fff;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛍️ Recommender System Inference App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Get personalized item recommendations from our best model</h4>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Model Info")
    st.write(f"Model Checkpoint: `{MODEL_CHECKPOINT_PATH}`")
    st.write(f"Inference Data: `{INFERENCE_DATA_DIR}`")

user_encoder, item_encoder, idx_to_item_id, model_params = load_inference_data()
model, device = load_model(model_params, MODEL_CHECKPOINT_PATH)

if model and user_encoder and model_params:
    with st.sidebar:
        st.success("Model and data loaded successfully!")
        st.write(f"Device: {device}")
        st.write(f"Num Users: {model_params['num_users']}")
        st.write(f"Num Items: {model_params['num_items']}")
        st.write(f"Embedding Dim: {model_params['embedding_dim']}")

    st.subheader("Get Recommendations")
    st.markdown(":bust_in_silhouette: <span style='color:#4CAF50'>Enter a User ID to get recommendations</span>", unsafe_allow_html=True)
    user_id_input = st.text_input("User ID", help="Enter an original User ID that was present during training.")
    k_input = st.slider("Number of recommendations (K)", min_value=1, max_value=50, value=DEFAULT_K)
    recommend_col, _ = st.columns([1, 2])
    with recommend_col:
        recommend_clicked = st.button("Recommend", use_container_width=True)
    if recommend_clicked:
        if not user_id_input:
            st.warning("Please enter a User ID.")
        else:
            try:
                try:
                    user_id_orig = int(user_id_input)
                except ValueError:
                    user_id_orig = user_id_input
                recommendations, error_msg = get_recommendations(
                    user_id_orig, model, device, k_input, user_encoder, item_encoder, idx_to_item_id
                )
                if error_msg:
                    st.error(error_msg)
                elif recommendations:
                    st.success(f"Top {k_input} recommendations for User ID {user_id_orig}:")
                    rec_df = pd.DataFrame({"Rank": np.arange(1, len(recommendations)+1), "Recommended Item ID": recommendations})
                    st.table(rec_df)
                else:
                    st.info("Could not generate recommendations.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.error("Failed to load model or inference data. Cannot proceed.")
