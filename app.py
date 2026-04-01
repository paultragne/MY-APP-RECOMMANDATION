import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
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

    /* même hauteur pour toutes les cartes */
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

/* Barre de recherche (visuelle seulement) */
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

/* Boutons Streamlit par défaut en orange */
.stButton > button {
    background-color: #FF9900;
    color: #FFFFFF;
    border-radius: 4px;
    border: none;
    width: 100%;
    transition: background-color 0.2s ease;
}
.stButton > button:hover {
    background-color: #CC7A00;
}

</style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)

st.markdown("""
<div class="search-bar">
    <input type="text" class="search-input" placeholder="Search Amazon Beauty...">
    <div class="search-button">🔍 Search</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Welcome back! Explore your personalized recommendations below.")

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
    scores = pd.Series(0, index=rating_matrix_cf.columns, dtype=float)

    for item, rating in rated_items.items():
        sims = item_similarity[item].drop(item)
        top_items = sims.sort_values(ascending=False).head(k)
        scores[top_items.index] += top_items * rating
    scores[rated_items.index] = -np.inf
    return scores.sort_values(ascending=False).head(100)

def recommend_user_based(user_id, top_n=5, k=30):
    sim_scores = user_similarity[user_id].drop(user_id)
    top_users = sim_scores.sort_values(ascending=False).head(k)
    weighted_ratings = rating_matrix_cf.loc[top_users.index].T.dot(top_users)
    scores = weighted_ratings / top_users.sum()
    user_rated = rating_matrix_cf.loc[user_id]
    scores[user_rated > 0] = -np.inf
    return scores.sort_values(ascending=False).head(100)

def recommend_hybrid(user_id, top_n=5):
    item_scores = recommend_item_based(user_id)
    user_scores = recommend_user_based(user_id)
    
    def normalize(series):
        valid_series = series[series != -np.inf]
        if valid_series.empty or valid_series.max() == valid_series.min():
            return series
        return (series - valid_series.min()) / (valid_series.max() - valid_series.min())

    norm_item = normalize(item_scores)
    norm_user = normalize(user_scores)
    
    pop_scores = popularity_model["avg_popularity"].head(100)
    norm_pop = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min()) if pop_scores.max() != pop_scores.min() else pop_scores

    combined_scores = pd.Series(0.0, index=rating_matrix_cf.columns)
    
    combined_scores[norm_item.index] += norm_item * 0.40
    combined_scores[norm_user.index] += norm_user * 0.20
    combined_scores[norm_pop.index] += norm_pop * 0.40

    bought_mask = (item_scores == -np.inf) | (user_scores == -np.inf)
    bought_items = bought_mask[bought_mask].index
    combined_scores[bought_items] = -np.inf

    return combined_scores.sort_values(ascending=False).head(100).index.tolist()


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


st.sidebar.header("👤 Your Amazon Account")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choose a User Profile:", users_list)


if selected_user:
    
    # 🛒 HISTORIQUE D'ACHATS
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


    # --- 1. HYBRIDE (« POUR VOUS ») --- EN PREMIERE POSITION
    st.subheader("🌟 For You (Personalized Mix)")

    all_hybrid_recs = recommend_hybrid(selected_user)
    filtered_hybrid_recs = [pid for pid in all_hybrid_recs if pid not in st.session_state.disliked_products]
    current_hybrid_recs = filtered_hybrid_recs[:5]

    col_hyb = st.columns(5)
    for i, pid in enumerate(current_hybrid_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        img_b64 = get_image_for_product(pid)

        with col_hyb[i]:
            st.markdown(f"""
            <div class="product-card">
                <img src="{img_b64}" class="product-img">
                <p><strong>Recommended For You</strong><br>{p_name[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Like", key=f"like_hyb_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")
            with col2:
                if st.button("Dislike", key=f"dislike_hyb_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

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

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Like", key=f"like_pop_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")
            with col2:
                if st.button("Dislike", key=f"dislike_pop_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

    st.divider()


    # --- 3. ITEM-BASED ---
    if last_purchased_name:
        contextual_header = f'Because of your purchases "{last_purchased_name[:30]}..."'
    else:
        contextual_header = "Because of your purchases"

    st.subheader(contextual_header)

    all_item_recs = recommend_item_based(selected_user).index.tolist()
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

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Like", key=f"like_item_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")
            with col2:
                if st.button("Dislike", key=f"dislike_item_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()

    st.divider()


    # --- 4. USER-BASED ---
    st.subheader("Customers also shopped for")

    all_user_recs = recommend_user_based(selected_user).index.tolist()
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
                <p><strong>People Like You Bought</strong><br>{p_name[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Like", key=f"like_user_{pid}", use_container_width=True):
                    st.toast("Saved to your list!")
            with col2:
                if st.button("Dislike", key=f"dislike_user_{pid}", use_container_width=True):
                    st.session_state.disliked_products.add(pid)
                    st.rerun()
