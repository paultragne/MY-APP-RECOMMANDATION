import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 1. PAGE CONFIGURATION & UI THEMING (AMAZON DA)
# ==========================================
st.set_page_config(
    page_title="Amazon Beauty Recommender",
    page_icon="🛍️",
    layout="wide"
)

# 🎨 APPLICATION DES COULEURS AMAZON (Noir, Orange, Blanc)
# Nous utilisons du CSS pour injecter les couleurs directement.
st.markdown("""
<style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #FFFFFF; /* White background */
        color: #111111; /* Amazon dark text */
    }

    /* Titles and Headings (Dark text) */
    h1, h2, h3, h3, h5, h6 {
        color: #111111 !important;
    }

    /* Sidebar styling (Dark background) */
    [data-testid="stSidebar"] {
        background-color: #232F3E; /* Amazon dark blue/black sidebar */
        color: #FFFFFF;
    }
    
    /* Text inside sidebar */
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Selectbox inside sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #111111; /* Dropdown text dark */
    }

    /* Recommendation boxes custom style (Cards) */
    .stInfo, .stSuccess, .stWarning {
        background-color: #F8F8F8; /* Light gray card background */
        border: 1px solid #DDDDDD; /* Subtle border */
        border-radius: 4px;
        color: #111111 !important;
        padding: 1rem;
    }
    
    /* Customization of the info, success, warning colors to match Amazon Orange */
    .stInfo, .stSuccess, .stWarning {
        border-top: 4px solid #FF9900 !important; /* Amazon Orange line */
    }

    /* Button and interaction colors (Amazon Orange) */
    .stButton>button {
        background-color: #FF9900;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #CC7A00; /* Darker orange on hover */
    }
</style>
""", unsafe_allow_html=True)

# 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Main Title)
st.title("🛍️ Amazon Beauty Recommendation System")
st.markdown("Enter a User ID in the sidebar to see personalized beauty recommendations.")

# ==========================================
# 📥 2. DATA LOADING
# ==========================================
# 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Data Loading Texts)
@st.cache_data
def load_data():
    # Reads the Excel file and keeps only 2500 rows to save memory
    df = pd.read_excel("Group3_Cleaned.xlsx", engine="openpyxl") 
    df = df.head(2500)
    return df
   
try:
    data_clean = load_data()
except Exception as e:
    # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Error Message)
    st.error(f"Error loading the database: {e}")
    st.stop()

# ==========================================
# 🧮 3. SIMILARITY MATRIX CALCULATIONS
# ==========================================
@st.cache_data
def prepare_matrices(df):
    rating_matrix = df.pivot_table(index="UserId", columns="ProductId", values="Rating").fillna(0)
    
    # User Similarity (User-Based)
    user_sim = cosine_similarity(rating_matrix)
    user_similarity = pd.DataFrame(user_sim, index=rating_matrix.index, columns=rating_matrix.index)
    
    # Item Similarity (Item-Based)
    item_sim = cosine_similarity(rating_matrix.T)
    item_similarity = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
    
    return rating_matrix, user_similarity, item_similarity

rating_matrix_cf, user_similarity, item_similarity = prepare_matrices(data_clean)

# Popularity Model Preparation (last 2 years)
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

# Item-Based Engine (Products similar to past purchases)
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

# User-Based Engine (Products liked by similar people)
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

# 🎨 Configuration CSS pour l'alignement parfait (CARRÉ)
# object-fit: cover for scaling without distortion, height is fixed to 200px
style_html_card = 'width:100%; height:200px; object-fit:cover; border-radius:4px; border: 1px solid #DDDDDD;'

# 🎨 Configuration CSS pour l'historique (plus petit)
style_html_history = 'width:100%; height:120px; object-fit:cover; border-radius:4px; border: 1px solid #DDDDDD;'


# 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Sidebar & Selectbox)
st.sidebar.header("👤 Your Amazon Account")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choose or paste a User ID:", users_list)

if selected_user:
    
    # 🎨 AJOUT DE PHOTOS DANS L'HISTORIQUE (Purchased products)
    st.subheader("🛒 Based on your purchase history")
    watched = data_clean[data_clean["UserId"] == selected_user][["ProductId", "product_name"]].drop_duplicates()
    
    # Layout the already purchased products as columns with images
    cols_purchased = st.columns(3)
    # Get up to 3 purchased products
    purchased_items = watched.head(3).iterrows()
    last_purchased_name = "" # Store the last name to use in modified item-based recommendation

    for i, row_data in enumerate(purchased_items):
        idx, row = row_data
        p_name = row['product_name']
        last_purchased_name = p_name # Update last_purchased_name

        with cols_purchased[i]:
            # 🔥 Utilisation d'images générées Picsum pour l'historique (carrées et alignées)
            # random=hist{i} force une image différente pour chaque achat
            url_generated = f"https://picsum.photos/300/300?random=hist{i}"
            st.markdown(
                f'<img src="{url_generated}" style="{style_html_history}">', 
                unsafe_allow_html=True
            )
            # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Purchased Text)
            st.markdown(f'<p style="color:#111111; font-size: 0.85rem; margin-top:5px;"><strong>Purchased</strong><br>{p_name[:60]}...</p>', unsafe_allow_html=True)
        
    st.divider()

    # --- LIGNE 1 : POPULARITÉ (Vrai texte Amazon + Couleur Orange) ---
    # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Popularity Header)
    st.subheader("🔥 Top 5 Best Sellers in Beauty (Popularité)")
    pop_recs = popularity_model.head(5).index.tolist()
    
    col_pop = st.columns(5)
    for i, pid in enumerate(pop_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_pop[i]:
            # 🔥 Utilisation d'images générées Picsum pour les recommandations (carrées et alignées)
            url_generated = f"https://picsum.photos/300/300?random=pop{i}"
            st.markdown(
                f'<img src="{url_generated}" style="{style_html_card}">', 
                unsafe_allow_html=True
            )
            # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Top Text)
            # The .stInfo style is customized to Amazon colors at the top.
            st.info(f"**Best Seller {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 2 : ITEM-BASED (Contextuelle "parce que vous avez acheté") ---
    # 🎨 RECOMMANDATION CONTEXTUELLE (Because you bought...)
    # We create the dynamic English header.
    if last_purchased_name:
        contextual_header = f'🎯 Items similar to "{last_purchased_name[:30]}..." (Item-Based)'
    else:
        contextual_header = "🎯 Recommended based on your purchases (Item-Based)"
        
    st.subheader(contextual_header)
    item_recs = recommend_item_based(selected_user)
    
    col_item = st.columns(5)
    for i, pid in enumerate(item_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_item[i]:
            # 🔥 Utilisation d'images générées Picsum
            url_generated = f"https://picsum.photos/300/300?random=item{i}"
            st.markdown(
                f'<img src="{url_generated}" style="{style_html_card}">', 
                unsafe_allow_html=True
            )
            # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Recommended Text)
            # St.success customized to Amazon Orange at the top.
            st.success(f"**Similar {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 3 : USER-BASED (Traductions) ---
    # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (User-Based Header)
    st.subheader("💡 You might also like... (User-Based)")
    user_recs = recommend_user_based(selected_user)
    
    col_user = st.columns(5)
    for i, pid in enumerate(user_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_user[i]:
            # 🔥 Utilisation d'images générées Picsum
            url_generated = f"https://picsum.photos/300/300?random=user{i}"
            st.markdown(
                f'<img src="{url_generated}" style="{style_html_card}">', 
                unsafe_allow_html=True
            )
            # 🎨 TRADUCTION DES TEXTES EN ANGLAIS (Recommended Text)
            # St.warning customized to Amazon Orange at the top.
            st.warning(f"**For You {i+1}**\n\n{p_name[:60]}...")
