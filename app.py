import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import random
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 1. PAGE CONFIGURATION & UI THEMING
# ==========================================
st.set_page_config(
    page_title="Amazon.com : Beauty & Personal Care",
    page_icon="🛍️",
    layout="wide"
)

# --- INITIALISATION DES VARIABLES DE SESSION (POUR LE VOTE) ---
if "disliked_products" not in st.session_state:
    st.session_state.disliked_products = set()

# --- CSS AMÉLIORÉ ---
st.markdown("""
<style>

.stApp { background-color: #FFFFFF; color: #111111; }
h1, h2, h3 { color: #111111 !important; }

/* Sidebar Amazon */
[data-testid="stSidebar"] { background-color: #232F3E; color: #FFFFFF; }
[data-testid="stSidebar"] .stMarkdown { color: #FFFFFF; }

/* Cartes produits améliorées */
.product-card {
    background-color: #F8F8F8;
    border: 1px solid #DDDDDD;
    border-radius: 6px;
    padding: 1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    border-top: 4px solid #FF9900 !important;

    /* --- même hauteur pour toutes les cartes --- */
    height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.product-card:hover {
    transform: scale(1.03);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Images */
.product-img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #DDDDDD;
}

.history-img {
    width: 100%;
    height: 120px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #DDDDDD;
}

/* Barre de recherche */
.search-bar {
    display: flex;
    margin-bottom: 2rem;
}
.search-input {
    width: 80%;
    padding: 10px;
    border: 1px solid #DDDDDD;
    border-right: none;
    border-radius: 4px 0 0 4px;
}
.search-button {
    width: 20%;
    padding: 10px;
    background-color: #FF9900;
    color: white;
    text-align: center;
    font-weight: bold;
    border-radius: 0 4px 4px 0;
    cursor: pointer;
}

/* --- BOUTONS LIKE / DISLIKE PREMIUM --- */

.vote-container {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
}

.vote-btn {
    width: 48%;
    padding: 8px 0;
    border-radius: 6px;
    text-align: center;
    font-weight: bold;
    cursor: pointer;
    border: none;
    font-size: 15px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

/* Bouton LIKE */
.vote-btn-like {
    background-color: #FF9900;
    color: white;
}
.vote-btn-like::before {
    content: "❤️ ";
    font-size: 16px;
}
.vote-btn-like:hover {
    background-color: #CC7A00;
    transform: scale(1.05);
    box-shadow: 0px 3px 8px rgba(0,0,0,0.2);
}

/* Bouton DISLIKE */
.vote-btn-dislike {
    background-color: #333333;
    color: white;
}
.vote-btn-dislike::before {
    content: "❌ ";
    font-size: 16px;
}
.vote-btn-dislike:hover {
    background-color: #111111;
    transform: scale(1.05);
    box-shadow: 0px 3px 8px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 📥 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel("Group3_Cleaned.xlsx", engine="openpyxl") 
    df = df.head(2500)
    return df

try:
    data_clean = load_data()
except Exception as e:
    st.error(f"Error loading database: {e}")
    st.stop()

# ==========================================
# 🧮 3. SIMILARITY MATRIX CALCULATIONS
# ==========================================
@st.cache_data
def prepare_matrices(df):
    rating_matrix = df.pivot_table(index="UserId", columns="ProductId", values="Rating").fillna(0)
    user_sim = cosine_similarity(rating_matrix)
    user_similarity = pd.DataFrame(user_sim, index=rating_matrix.index, columns=rating_matrix.index)
    item_sim = cosine_similarity(rating_matrix.T)
    item_similarity = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
    return rating_matrix, user_similarity, item_similarity

rating_matrix_cf, user_similarity, item_similarity = prepare_matrices(data_clean)

latest_date = pd.to_datetime(data_clean["Timestamp_Converted"]).max()
two_years_ago = latest_date - pd.DateOffset(years=2)
data_recent = data_clean[pd.to_datetime(data_clean["Timestamp_Converted"]) >= two_years_ago].copy()

popularity_model = data_recent.groupby("ProductId").agg(
    avg_popularity=("Rating", "mean"),
    n_rating=("Rating", "count")
)
popularity_model = popularity_model[popularity_model["n_rating"] >= 2].sort_values("avg_popularity", ascending=False)

# ==========================================
# 🧠 4. RECOMMENDATION ENGINES
# ==========================================
def recommend_item_based(user_id, top_n=5, k=30):
    user_ratings = rating_matrix_cf.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    scores = pd.Series(0, index=rating_matrix_cf.columns)
    for item, rating in rated_items.items():
        sims = item_similarity[item].drop(item)
        top_items = sims.sort_values(ascending=False).head(k)
        scores[top_items.index] += top_items * rating
    scores[rated_items.index] = -np.inf
    return scores.sort_values(ascending=False).head(100).index.tolist()

def recommend_user_based(user_id, top_n=5, k=30):
    sim_scores = user_similarity[user_id].drop(user_id)
    top_users = sim_scores.sort_values(ascending=False).head(k)
    weighted_ratings = rating_matrix_cf.loc[top_users.index].T.dot(top_users)
    scores = weighted_ratings / top_users.sum()
    user_rated = rating_matrix_cf.loc[user_id]
    scores[user_rated > 0] = -np.inf
    return scores.sort_values(ascending=False).head(100).index.tolist()

# ==========================================
# 🖥️ 5. VISUAL INTERFACE (UI)
# ==========================================
chemin_dossier_images = "image_produits"
banque_images_locales = []

try:
    if os.path.exists(chemin_dossier_images):
        for fichier in os.listdir(chemin_dossier_images):
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
                banque_images_locales.append(os.path.join(chemin_dossier_images, fichier))
except Exception as e:
    st.error(f"Error reading image_produits folder: {e}")

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            return f"data:{mime};base64,{encoded}"
    except:
        return None

def get_image_for_product(product_id):
    if not banque_images_locales:
        return None
    index_image = hash(product_id) % len(banque_images_locales)
    return get_base64_image(banque_images_locales[index_image])

# Styles HTML
style_html_card = 'product-img'
style_html_history = 'history-img'

st.sidebar.header("👤 Your Amazon Account")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choose a User Profile:", users_list)


if selected_user:
    
    # 🛒 1. HISTORIQUE D'ACHATS
    st.subheader("🛒 Based on your recent purchases")
    watched = data_clean[data_clean["UserId"] == selected_user][["ProductId", "product_name"]].drop_duplicates()
    
    cols_purchased = st.columns(3)
    purchased_items = watched.head(3).iterrows()
    last_purchased_name = ""

    for i, row_data in enumerate(purchased_items):
        idx, row = row_data
        p_name = row['product_name']
        last_purchased_name = p_name

        with cols_purchased[i]:
            img_b64 = get_image_for_product(row['ProductId'])
            if img_b64:
                st.markdown(f'<img src="{img_b64}" class="history-img">', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#111111; font-size: 0.85rem; margin-top:5px;"><strong>Purchased Item</strong><br>{p_name[:50]}...</p>',
                unsafe_allow_html=True
            )
        
    st.divider()

    # --- 2. POPULARITÉ ---
    st.subheader("Top 5 Best Sellers in Beauty")

    all_pop_recs = popularity_model.index.tolist()
    filtered_pop_recs = [pid for pid in all_pop_recs if pid not in st.session_state.disliked_products]
    current_pop_recs = filtered_pop_recs[:5]

    col_pop = st.columns(5)
    for i, pid in enumerate(current_pop_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        img_b64 = get_image_for_product(pid)

        with col_pop[i]:
            st.markdown(f"""
            <div class="product-card">
                <img src="{img_b64}" class="product-img">
                <p><strong>Best Seller</strong><br>{p_name[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Boutons Like / Dislike premium ---
            st.markdown('<div class="vote-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Like", key=f"like_pop_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")

            with col2:
                if st.button("Dislike", key=f"dislike_pop_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- 3. ITEM-BASED ---
    if last_purchased_name:
        contextual_header = f'Because of your purchases "{last_purchased_name[:30]}..."'
    else:
        contextual_header = "Because of your purchases"

    st.subheader(contextual_header)

    all_item_recs = recommend_item_based(selected_user)
    filtered_item_recs = [pid for pid in all_item_recs if pid not in st.session_state.disliked_products]
    current_item_recs = filtered_item_recs[:5]

    col_item = st.columns(5)
    for i, pid in enumerate(current_item_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        img_b64 = get_image_for_product(pid)

        with col_item[i]:
            st.markdown(f"""
            <div class="product-card">
                <img src="{img_b64}" class="product-img">
                <p><strong>Similar Match</strong><br>{p_name[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Boutons Like / Dislike premium ---
            st.markdown('<div class="vote-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Like", key=f"like_item_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")

            with col2:
                if st.button("Dislike", key=f"dislike_item_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- 4. USER-BASED ---
    st.subheader("Customers also shopped for")

    all_user_recs = recommend_user_based(selected_user)
    filtered_user_recs = [pid for pid in all_user_recs if pid not in st.session_state.disliked_products]
    current_user_recs = filtered_user_recs[:5]

    col_user = st.columns(5)
    for i, pid in enumerate(current_user_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        img_b64 = get_image_for_product(pid)

        with col_user[i]:
            st.markdown(f"""
            <div class="product-card">
                <img src="{img_b64}" class="product-img">
                <p><strong>For You</strong><br>{p_name[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Boutons Like / Dislike premium ---
            st.markdown('<div class="vote-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Like", key=f"like_user_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")

            with col2:
                if st.button("Dislike", key=f"dislike_user_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
# (Fin du if selected_user:)
