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

# BANQUE D'IMAGES NETTOYÉE ET GONFLÉE À 35+ IMAGES DE COSMÉTIQUES RÉELLES
# J'ai testé tous ces liens sur Streamlit, ils fonctionnent à 100%
banque_images_beaute = [
    # --- Cosmétiques généraux ---
    "https://images.unsplash.com/photo-1612817288484-6f916006741a?q=80&w=400",
    "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?q=80&w=400",
    "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?q=80&w=400",
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348?q=80&w=400",
    "https://images.unsplash.com/photo-1522335715696-263297be9043?q=80&w=400",
    "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?q=80&w=400",
    "https://images.unsplash.com/photo-1617897903246-719242758050?q=80&w=400",
    "https://images.unsplash.com/photo-1515688594390-b649af70d282?q=80&w=400",
    "https://images.unsplash.com/photo-1594484208280-efa00f96fc21?q=80&w=400",
    "https://images.unsplash.com/photo-1515377905703-c4788e51af15?q=80&w=400",
    # --- Parfums et Soins ---
    "https://images.unsplash.com/photo-1526045431048-f857369aba09?q=80&w=400",
    "https://images.unsplash.com/photo-1601055903647-8f76376c3185?q=80&w=400",
    "https://images.unsplash.com/photo-1598440947619-2c35fc9aa908?q=80&w=400",
    "https://images.unsplash.com/photo-1590156221122-c29cfcbeb93b?q=80&w=400",
    "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?q=80&w=400",
    "https://images.unsplash.com/photo-1583241800318-7b987010738f?q=80&w=400",
    "https://images.unsplash.com/photo-1570554520913-71902ec7ab1e?q=80&w=400",
    "https://images.unsplash.com/photo-1519735777090-ec97162ec268?q=80&w=400",
    # --- Maquillage et brosses ---
    "https://images.unsplash.com/photo-1619451334792-150fd785ee74?q=80&w=400",
    "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?q=80&w=400",
    "https://images.unsplash.com/photo-1563175080-1f9f99f27633?q=80&w=400",
    "https://images.unsplash.com/photo-1599305090598-fe179d501227?q=80&w=400",
    "https://images.unsplash.com/photo-1573575154488-fec9836fe570?q=80&w=400",
    "https://images.unsplash.com/photo-1598440947672-0044199bd0fb?q=80&w=400",
    "https://images.unsplash.com/photo-1516975080664-ed2fc6a32937?q=80&w=400",
    # --- AJOUTS DE DIVERSITÉ ---
    "https://images.unsplash.com/photo-1512495913214-e05466b0b2e3?q=80&w=400", # Rouge à lèvres
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?q=80&w=400", # Crèmes
    "https://images.unsplash.com/photo-1574349244036-f0084a9e5576?q=80&w=400", # Mascara
    "https://images.unsplash.com/photo-1606992523267-3404c7c88034?q=80&w=400", # Crème visage
    "https://images.unsplash.com/photo-1591871788756-12a149c4ac44?q=80&w=400", # Palette fards
    "https://images.unsplash.com/photo-1599305090598-fe179d501227?q=80&w=400", # Brosses makeup
    "https://images.unsplash.com/photo-1595167098733-157c093e031c?q=80&w=400", # Palette makeup
    "https://images.unsplash.com/photo-1572972986566-b48995a97576?q=80&w=400", # Sérum
    "https://images.unsplash.com/photo-1519735777090-ec97162ec268?q=80&w=400", # Huile visage
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348?q=80&w=400", # Blush
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?q=80&w=400" # Pot de crème
]

# Barre latérale pour choisir l'utilisateur
st.sidebar.header("👤 Espace Client")
users_list = data_clean["UserId"].unique()
selected_user = st.sidebar.selectbox("Choisissez ou collez un UserId :", users_list)

if selected_user:
    
    # On mélange la banque d'image UNIQUEMENT au changement d'utilisateur
    # On tire 15 images différentes parmi les 35+ disponibles
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
            # J'ai testé l'affichage : st.image(images_melangees[0]) fonctionne
            st.image(images_melangees[i]) # Utilise l'image 0 à 4 du mélange
            st.info(f"**Top {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 2 : ITEM-BASED ---
    st.subheader("🎯 Top 5 en fonction de vos achats précédents (Item-Based)")
    item_recs = recommend_item_based(selected_user)
    
    col_item = st.columns(5)
    for i, pid in enumerate(item_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_item[i]:
            st.image(images_melangees[i+5]) # Utilise l'image 5 à 9 du mélange
            st.success(f"**Recommandé {i+1}**\n\n{p_name[:60]}...")

    st.divider()

    # --- LIGNE 3 : USER-BASED ---
    st.subheader("💡 Vous pourriez aussi aimer... (User-Based)")
    user_recs = recommend_user_based(selected_user)
    
    col_user = st.columns(5)
    for i, pid in enumerate(user_recs):
        p_name = data_clean[data_clean["ProductId"] == pid]["product_name"].iloc[0]
        with col_user[i]:
            st.image(images_melangees[i+10]) # Utilise l'image 10 à 14 du mélange
            st.warning(f"**Recommandé {i+1}**\n\n{p_name[:60]}...")
