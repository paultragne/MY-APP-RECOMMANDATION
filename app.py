import streamlit as st
import pandas as pd
import numpy as np
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

# CSS pour le style Amazon (Noir, Orange, Blanc) et alignement des boutons
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111111; }
    h1, h2, h3 { color: #111111 !important; }
    
    /* Sidebar Amazon */
    [data-testid="stSidebar"] { background-color: #232F3E; color: #FFFFFF; }
    [data-testid="stSidebar"] .stMarkdown { color: #FFFFFF; }
    [data-testid="stSidebar"] [data-baseweb="select"] { color: #111111; }

    /* Cartes produits */
    .stInfo, .stSuccess, .stWarning {
        background-color: #F8F8F8;
        border: 1px solid #DDDDDD;
        border-radius: 4px;
        color: #111111 !important;
        padding: 1rem;
        border-top: 4px solid #FF9900 !important;
    }

    /* Boutons Amazon Orange */
    .stButton>button {
        background-color: #FF9900;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #CC7A00; }
    
    /* Barre de recherche Amazon */
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
</style>
""", unsafe_allow_html=True)

# --- BANNIÈRE AMAZON RÉELLE ---
st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)

# --- MODIFICATION 1 : BARRE DE RECHERCHE FACTICE (STYLE AMAZON) ---
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
    scores = pd.Series(0, index=rating_matrix_cf.columns)
    for item, rating in rated_items.items():
        sims = item_similarity[item].drop(item)
        top_items = sims.sort_values(ascending=False).head(k)
        scores[top_items.index] += top_items * rating
    scores[rated_items.index] = -np.inf
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_user_based(user_id, top_n=5, k=30):
    sim_scores = user_similarity[user_id].drop(user_id)
    top_users = sim_scores.sort_values(ascending=False).head(k)
    weighted_ratings = rating_matrix_cf.loc[top_users.index].T.dot(top_users)
    scores = weighted_ratings / top_users.sum()
    user_rated = rating_matrix_cf.loc[user_id]
    scores[user_rated > 0] = -np.inf
    return scores.sort_values(ascending=False).head(top_n).index.tolist()


# ==========================================
# 🖥️ 5. VISUAL INTERFACE (UI)
# ==========================================

style_html_card = 'width:100%; height:180px; object-fit:cover; border-radius:4px; border: 1px solid #DDDDDD;'
style_html_history = 'width:100%; height:120px; object-fit:cover; border-radius:4px; border: 1px solid #DDDDDD;'

st.sidebar.header("👤 Your Amazon Account")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choose a User Profile:", users_list)

if selected_user:
    
    # 🛒 HISTORIQUE D'ACHATS VISUEL
    st.subheader("🛒 Based on your purchase history")
    watched = data_clean[data_clean["UserId"] == selected_user][["ProductId", "product_name"]].drop_duplicates()
    
    cols_purchased = st.columns(3)
    purchased_items = watched.head(3).iterrows()
    last_purchased_name = ""

    for i, row_data in enumerate(purchased_items):
        idx, row = row_data
        p_name = row['product_name']
        last_purchased_name = p_name

        with cols_purchased[i]:
            url_generated = f"https://picsum.photos/300/300?random=hist{i}"
            st.markdown(f'<img src="{url_generated}" style="{style_html_history}">', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#111111; font-size: 0.85rem; margin-top:5px;"><strong>Purchased Item</strong><br>{p_name[:50]}...</p>', unsafe_allow_html=True)
        
    st.divider()

    # --- MODIFICATION 2 : LIGNE 1 POPULARITÉ SANS LE MOT POPULARITÉ ET AVEC LES BOUTONS 👍/👎 ---
    st.subheader("🔥 Top 5 Best Sellers in Beauty")
    
    # On filtre les produits populaires : on exclut ceux que l'utilisateur a déjà "disliké"
    all_pop_recs = popularity_model.index.tolist()
    filtered_pop_recs = [pid for pid in all_pop_recs if pid not in st.session_state.disliked_products]
    
    # On prend les 5 premiers restants
    current_pop_recs = filtered_pop_recs[:5]
    
    col_pop = st.columns(5)
    for i, pid in enumerate(current_pop_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_pop[i]:
            url_generated = f"https://picsum.photos/300/300?random=pop{i}"
            st.markdown(f'<img src="{url_generated}" style="{style_html_card}">', unsafe_allow_html=True)
            st.info(f"**Best Seller {i+1}**\n\n{p_name[:50]}...")
            
            # --- Système de Vote (Boutons miniatures côte à côte) ---
            vote_col1, vote_col2 = st.columns(2)
            with vote_col1:
                if st.button(f"👍", key=f"like_pop_{pid}"):
                    st.toast(f"Saved to your wishlist!") # Notification légère en bas à droite
            with vote_col2:
                if st.button(f"👎", key=f"dislike_pop_{pid}"):
                    st.session_state.disliked_products.add(pid) # On l'ajoute aux bannis
                    st.rerun() # On recharge instantanément la page

    st.divider()

    # --- MODIFICATION 3 : LIGNE 2 SANS LE MOT ITEM-BASED ---
    if last_purchased_name:
        contextual_header = f'🎯 Items similar to "{last_purchased_name[:30]}..."'
    else:
        contextual_header = "🎯 Recommended based on your purchases"
        
    st.subheader(contextual_header)
    item_recs = recommend_item_based(selected_user)
    
    col_item = st.columns(5)
    for i, pid in enumerate(item_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_item[i]:
            url_generated = f"https://picsum.photos/300/300?random=item{i}"
            st.markdown(f'<img src="{url_generated}" style="{style_html_card}">', unsafe_allow_html=True)
            st.success(f"**Similar Match**\n\n{p_name[:50]}...")

    st.divider()

    # --- MODIFICATION 4 : LIGNE 3 SANS LE MOT USER-BASED ---
    st.subheader("💡 Customers also shopped for")
    user_recs = recommend_user_based(selected_user)
    
    col_user = st.columns(5)
    for i, pid in enumerate(user_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_user[i]:
            url_generated = f"https://picsum.photos/300/300?random=user{i}"
            st.markdown(f'<img src="{url_generated}" style="{style_html_card}">', unsafe_allow_html=True)
            st.warning(f"**For You**\n\n{p_name[:50]}...")
