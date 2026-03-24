import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 1. CONFIGURATION DE LA PAGE VISUELLE
# ==========================================
st.set_page_config(
    page_title="Recommandation Amazon Beauty",
    page_icon="🛍️",
    layout="wide"
)

# Titre principal de l'application
st.title("🛍️ Système de Recommandation Amazon Beauty")
st.markdown("Rentrez un numéro d'utilisateur (User ID) pour voir ses recommandations personnalisées.")

# ==========================================
# 📥 2. CHARGEMENT DE LA BASE DE DONNÉES
# ==========================================
@st.cache_data
def load_data():
    # Lit le fichier Excel et ne garde que 2500 lignes pour économiser la mémoire
    df = pd.read_excel("Group3_Cleaned.xlsx", engine="openpyxl") 
    df = df.head(2500)
    return df
   
try:
    data_clean = load_data()
except Exception as e:
    st.error(f"Erreur de chargement de la base de données : {e}")
    st.stop()

# ==========================================
# 🧮 3. CALCULS DES MATRICES DE SIMILARITÉ
# ==========================================
@st.cache_data
def prepare_matrices(df):
    rating_matrix = df.pivot_table(index="UserId", columns="ProductId", values="Rating").fillna(0)
    
    # Similarité Utilisateur (User-Based)
    user_sim = cosine_similarity(rating_matrix)
    user_similarity = pd.DataFrame(user_sim, index=rating_matrix.index, columns=rating_matrix.index)
    
    # Similarité Produit (Item-Based)
    item_sim = cosine_similarity(rating_matrix.T)
    item_similarity = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
    
    return rating_matrix, user_similarity, item_similarity

rating_matrix_cf, user_similarity, item_similarity = prepare_matrices(data_clean)

# Préparation du modèle de Popularité (2 dernières années)
latest_date = pd.to_datetime(data_clean["Timestamp_Converted"]).max()
two_years_ago = latest_date - pd.DateOffset(years=2)
data_recent = data_clean[pd.to_datetime(data_clean["Timestamp_Converted"]) >= two_years_ago].copy()

popularity_model = data_recent.groupby("ProductId").agg(
    avg_popularity=("Rating", "mean"),
    n_rating=("Rating", "count")
)
popularity_model = popularity_model[popularity_model["n_rating"] >= 2].sort_values("avg_popularity", ascending=False)


# ==========================================
# 🧠 4. LES MOTEURS DE RECOMMANDATION
# ==========================================

# Moteur Item-Based (Produits similaires aux achats passés)
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

# Moteur User-Based (Produits aimés par des gens similaires)
def recommend_user_based(user_id, top_n=5, k=30):
    sim_scores = user_similarity[user_id].drop(user_id)
    top_users = sim_scores.sort_values(ascending=False).head(k)
    weighted_ratings = rating_matrix_cf.loc[top_users.index].T.dot(top_users)
    scores = weighted_ratings / top_users.sum()
    user_rated = rating_matrix_cf.loc[user_id]
    scores[user_rated > 0] = -np.inf
    return scores.sort_values(ascending=False).head(top_n).index.tolist()


# ==========================================
# 🖥️ 5. L'INTERFACE VISUELLE (CE QUE L'ON VOIT)
# ==========================================

import random # Outil pour le mélange aléatoire

# --- BANQUE GÉANTE DE 65+ IMAGES DE BEAUTÉ RÉELLES ET TESTÉES ---
banque_images_beaute = [
    "https://images.unsplash.com/photo-1612817288484-6f916006741a?w=400",
    "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400",
    "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=400",
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400",
    "https://images.unsplash.com/photo-1522335715696-263297be9043?w=400",
    "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400",
    "https://images.unsplash.com/photo-1617897903246-719242758050?w=400",
    "https://images.unsplash.com/photo-1515688594390-b649af70d282?w=400",
    "https://images.unsplash.com/photo-1594484208280-efa00f96fc21?w=400",
    "https://images.unsplash.com/photo-1515377905703-c4788e51af15?w=400",
    "https://images.unsplash.com/photo-1526045431048-f857369aba09?w=400",
    "https://images.unsplash.com/photo-1601055903647-8f76376c3185?w=400",
    "https://images.unsplash.com/photo-1598440947619-2c35fc9aa908?w=400",
    "https://images.unsplash.com/photo-1590156221122-c29cfcbeb93b?w=400",
    "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?w=400",
    "https://images.unsplash.com/photo-1583241800318-7b987010738f?w=400",
    "https://images.unsplash.com/photo-1570554520913-71902ec7ab1e?w=400",
    "https://images.unsplash.com/photo-1519735777090-ec97162ec268?w=400",
    "https://images.unsplash.com/photo-1619451334792-150fd785ee74?w=400",
    "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=400",
    "https://images.unsplash.com/photo-1563175080-1f9f99f27633?w=400",
    "https://images.unsplash.com/photo-1599305090598-fe179d501227?w=400",
    "https://images.unsplash.com/photo-1573575154488-fec9836fe570?w=400",
    "https://images.unsplash.com/photo-1598440947672-0044199bd0fb?w=400",
    "https://images.unsplash.com/photo-1516975080664-ed2fc6a32937?w=400",
    "https://images.unsplash.com/photo-1512495913214-e05466b0b2e3?w=400", 
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400", 
    "https://images.unsplash.com/photo-1574349244036-f0084a9e5576?w=400", 
    "https://images.unsplash.com/photo-1606992523267-3404c7c88034?w=400", 
    "https://images.unsplash.com/photo-1591871788756-12a149c4ac44?w=400", 
    "https://images.unsplash.com/photo-1595167098733-157c093e031c?w=400", 
    "https://images.unsplash.com/photo-1572972986566-b48995a97576?w=400", 
    "https://images.unsplash.com/photo-1512495518311-6b72183c5e00?w=400",
    "https://images.unsplash.com/photo-1631730486784-029911d96b01?w=400",
    "https://images.unsplash.com/photo-1614859324967-bdf411dfa467?w=400",
    "https://images.unsplash.com/photo-1608541737042-87a12275d313?w=400",
    "https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=400",
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400",
    "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=400",
    "https://images.unsplash.com/photo-1606992523267-3404c7c88034?w=400",
    "https://images.unsplash.com/photo-1591871788756-12a149c4ac44?w=400",
    "https://images.unsplash.com/photo-1631730486784-029911d96b01?w=400",
    "https://images.unsplash.com/photo-1612817288484-6f916006741a?w=400",
    "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400",
    "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400",
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400",
    "https://images.unsplash.com/photo-1512495518311-6b72183c5e00?w=400",
    "https://images.unsplash.com/photo-1614859324967-bdf411dfa467?w=400",
    "https://images.unsplash.com/photo-1631729371254-42c2892f0e6e?w=400",
    "https://images.unsplash.com/photo-1608541737042-87a12275d313?w=400",
    "https://images.unsplash.com/photo-1598454441315-1888b5b5c960?w=400",
    "https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=400"
]

# Barre latérale pour choisir l'utilisateur
st.sidebar.header("👤 Espace Client")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choisissez ou collez un UserId :", users_list)

if selected_user:
    
    # On tire 15 images aléatoires dans notre banque de beauté
    images_melangees = random.sample(banque_images_beaute, 15)
    
    # Historique des achats de l'utilisateur
    st.subheader("🛒 Produits déjà achetés par cet utilisateur")
    watched = data_clean[data_clean["UserId"] == selected_user][["ProductId", "product_name"]].drop_duplicates()
    for _, row in watched.head(3).iterrows():
        st.write(f"- {row['product_name']} (ID: {row['ProductId']})")
        
    st.divider()

    # --- LIGNE 1 : POPULARITÉ ---
    st.subheader("🔥 Top 5 des produits en vogue du moment (Popularité)")
    pop_recs = popularity_model.head(5).index.tolist()
    
    col_pop = st.columns(5)
    for i, pid in enumerate(pop_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_pop[i]:
            # 🔥 On utilise du HTML pour forcer la hauteur de l'image (250px) et l'alignement
            st.markdown(
                f'<img src="{images_melangees[i]}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">', 
                unsafe_allow_html=True
            )
            st.info(f"**Top {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 2 : ITEM-BASED ---
    st.subheader("🎯 Top 5 en fonction de vos achats précédents (Item-Based)")
    item_recs = recommend_item_based(selected_user)
    
    col_item = st.columns(5)
    for i, pid in enumerate(item_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_item[i]:
            st.markdown(
                f'<img src="{images_melangees[i+5]}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">', 
                unsafe_allow_html=True
            )
            st.success(f"**Recommandé {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 3 : USER-BASED ---
    st.subheader("💡 Vous pourriez aussi aimer... (User-Based)")
    user_recs = recommend_user_based(selected_user)
    
    col_user = st.columns(5)
    for i, pid in enumerate(user_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_user[i]:
            st.markdown(
                f'<img src="{images_melangees[i+10]}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">', 
                unsafe_allow_html=True
            )
            st.warning(f"**Recommandé {i+1}**\n\n{p_name[:60]}...")
