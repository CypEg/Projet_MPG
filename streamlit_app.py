# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:45:05 2023

@author: cypri
"""

#%% Importation des librairies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#%% Importation des données de chaque ligue
Ligue1_J9_raw = pd.read_csv("Data/Projet MPG - Ligue 1 J9.csv")
Ligue1_J11_raw = pd.read_csv("Data/Projet MPG - Ligue 1 J11.csv")
Ligue2_J11_raw = pd.read_csv("Data/Projet MPG - Ligue 2 J11.csv")
Ligue2_J13_raw = pd.read_csv("Data/Projet MPG - Ligue 2 J13.csv")
PremierLeague_J9_raw = pd.read_csv("Data/Projet MPG - Premier League J9.csv")
PremierLeague_J11_raw = pd.read_csv("Data/Projet MPG - Premier League J11.csv")
Liga_J10_raw = pd.read_csv("Data/Projet MPG - Liga J10.csv")
Liga_J12_raw = pd.read_csv("Data/Projet MPG - Liga J12.csv")
SerieA_J9_raw = pd.read_csv("Data/Projet MPG - Serie A J9.csv")
SerieA_J11_raw = pd.read_csv("Data/Projet MPG - Serie A J11.csv")

#%% Ajoute du titre et des pages 
st.title("Projet DataScientest : MPG")
st.sidebar.title("Sommaire")
pages=["Le Projet", 
       "Les Données", 
       "Visualisation 1ère approche", 
       "Visualisation 2ème approche",
       "Modélisation : Régression Linéaire",
       "Modélisation : Classification ----- 1ère partie",
       "Modélisation : Classification ----- 2ème partie",
       "Fonction : Mercato",
       "Fonction : Composition d'équipe"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.markdown("## Cyprien EGLY")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/cyprien-egly/)")

#%% Ajout de la présentation à la page Le Projet
if page == pages[0] : 
  st.write("## Présentation du projet")
  st.write("### Contexte")
  st.markdown("<div style='text-align: justify;'>Le projet est basé sur l’application de jeu de fantasy football “MPG” MonPetitGazon. Le but de ce jeu est de gagner des matchs de football virtuel grâce aux performances réelles des joueurs et aux notes qui leurs sont attribués à l'issu de chaque rencontre. Le joueur obtenant le score le plus élevé après l'addition des notes de ses footballeurs est déclaré vainqueur.</div>", unsafe_allow_html=True)
  st.write("### Objectifs")
  st.markdown("<div style='text-align: justify;'>L’objectif principal de ce projet est de proposer aux utilisateurs du jeu MPG des outils intelligents afin de les aider à constituer la meilleure équipe durant le mercato et à sélectionner la meilleure composition d'équipe avant chaque match.</div>", unsafe_allow_html=True)

#%% Ajout du contenu de la page Les Données
# Pré-processing des données
# Gestion des homonymes
Ligue1_J9_raw.loc[(Ligue1_J9_raw["Joueur"] == "Marquinhos") & (Ligue1_J9_raw["Club"] == "FC Nantes"), "Joueur"] = "Marquinhos Oliveira Alencar"
Ligue1_J11_raw.loc[(Ligue1_J9_raw["Joueur"] == "Marquinhos") & (Ligue1_J11_raw["Club"] == "FC Nantes"), "Joueur"] = "Marquinhos Oliveira Alencar"
PremierLeague_J9_raw.loc[(PremierLeague_J9_raw["Joueur"] == "Danilo") & (PremierLeague_J9_raw["Club"] == "Nottingham Forest"), "Joueur"] = "Danilo dos Santos de Oliveira"
PremierLeague_J11_raw.loc[(PremierLeague_J11_raw["Joueur"] == "Danilo") & (PremierLeague_J11_raw["Club"] == "Nottingham Forest"), "Joueur"] = "Danilo dos Santos de Oliveira"

PremierLeague_J9_raw.loc[(PremierLeague_J9_raw["Joueur"] == "Hwang Hee-Chan Hee-Chan"), "Joueur"] = "Hwang Hee-Chan"
PremierLeague_J11_raw.loc[(PremierLeague_J9_raw["Joueur"] == "Hwang Hee-Chan Hee-Chan"), "Joueur"] = "Hwang Hee-Chan"

# Ajout d'une colonne ligue pour chaque ligue
Ligue1_J9_raw["Ligue"] = "Ligue 1"
Ligue2_J11_raw["Ligue"] = "Ligue 2"
PremierLeague_J9_raw["Ligue"] = "Premier League"
Liga_J10_raw["Ligue"] = "Liga"
SerieA_J9_raw["Ligue"] = "Serie A"

# Sélection des colonnes essentielles à l'étude
essential_col = ["Joueur", "Poste", "Cote", "Var cote", "% achat", "% achat tour 1", "Q2 Toutes tailles", 'Q3 Toutes tailles', 'Q2 à 6 joueurs', 'Q3 à 6 joueurs',
                 'Q2 à 8 joueurs', 'Q3 à 8 joueurs', 'Q2 à 10 joueurs', 'Q3 à 10 joueurs', 'Note', 'Note série', 'Note 1 an', 'Note M11', "Nb match", "Nb match série",
                 "Nb match 1 an", 'But', "%Titu", "Temps", "Cleansheet",'But/Peno', 'But/Coup-franc', 'But/surface', 'Pass decis.', 'Occas° créée',
                 'Tirs', 'Tirs cadrés', 'Corner gagné', '%Passes', 'Ballons', 'Interceptions', 'Tacles', '%Duel', 'Fautes', 'But évité',
                 'Action stoppée', 'Poss Def', 'Poss Mil', 'Centres', 'Centres ratés', 'Dégagements', "But concédé", 'Ballon perdu', 'Passe>Tir', 'Passe perfo',
                 'Dépossédé', 'Plonge&stop', "Nb matchs gagnés", 'Erreur>But', "Diff de buts", 'Grosse occas manquée', 'Balle non rattrapée', "Bonus moy", "Malus moy", "Club",
                 "Ligue"]

# Concaténation des data frames dans all_players
all_players_raw = pd.concat([Ligue1_J9_raw[essential_col], 
                             Ligue2_J11_raw[essential_col], 
                             PremierLeague_J9_raw[essential_col], 
                             Liga_J10_raw[essential_col], 
                             SerieA_J9_raw[essential_col]], ignore_index=True)

# Gestion des NA et des valeurs des colonnes
# Création d'une copie de travail
all_players = all_players_raw

# Pour les colonnes % achat et % achat tour 1, remplacement par la moyenne
all_players['% achat'] = all_players['% achat'].fillna(all_players['% achat'].median())
all_players['% achat tour 1'] = all_players['% achat tour 1'].fillna(all_players['% achat tour 1'].median())

# Pour les autres colonnes, remplacement par 0
all_players = all_players.fillna(0)

# Pour la colonne %Titu, multiplication des valeurs par 100
all_players["%Titu"] = all_players["%Titu"] * 100

#%% Mise en page de la page Les Données
if page == pages[1] :
    st.write("## Les Données")
    st.write("Les données utilisées lors de ce projet proviennent du site [MPG stats](https://www.mpgstats.fr/).")
    st.markdown("<div style='text-align: justify;'>Les données d'origines se présentent sous la forme de tableau selon différentes information comme le tableau de la ligue, des tableaux sur les top joueurs ainsi que sur toutes les équipes. La première étape de la collecte des données a donc été de récupérer toutes les infos par équipe de chaque ligue pour obtenir le résultat suivant:</div>", unsafe_allow_html=True)
    st.write("")
    st.write(Ligue1_J9_raw.head(5))
    st.write("Soit un total de", len(Ligue1_J9_raw.columns), " colonnes.")
    st.markdown("<div style='text-align: justify;'>Une fois toutes les ligues regroupées, une première gestion des noms des joueurs a été faite afin d'éviter les homonynes ou les noms mal écrits. Un travail sur les colonnes a ensuite été effectué avec la création d'une colonne Ligue et une sélection de colonne essentielle à l'étude. Ainsi, toutes les colonnes résultants d'un calcul (moyenne, total etc...) ont été retirées. Le résultat nous donne un jeu de données de cette dimension: </div>", unsafe_allow_html=True)
    st.write("")
    st.write(all_players_raw.shape)
    st.markdown("<div style='text-align: justify;'>Les colonnes de notre dataset peuvent être classées en deux catégories :</div>", unsafe_allow_html=True)
    st.markdown("- 4 colonnes qualitatives")
    st.markdown("- 57 colonnes quantitatives")
    st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)   
    st.markdown("<div style='text-align: justify;'>Ces colonnes quantitatives peuvent à leur tout être classées en deux catégories :</div>", unsafe_allow_html=True)
    st.markdown("- 12 colonnes relatives à la valeur et l’achat des joueurs")
    st.markdown("- 45 colonnes relatives à des statistiques de match")    
    st.markdown("<div style='text-align: justify;'>Dû à ces distinctions, nos colonnes statistiques se retrouvent parfois avec un nombre élevé de valeurs manquantes.</div>", unsafe_allow_html=True)
    if st.checkbox("Afficher les NA") :
        st.dataframe(all_players_raw.isna().sum())
    st.markdown("<div style='text-align: justify;'>Elle représentent en fait des absences d'actions dans chaques catégories. Nous décidons alors de les remplacer par des zéros.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Enfin, la dernière étape de pré-processing de nos données a été la mise en valeur pourcentage de la colonne %Titu qui était à l'origine sous le format 0.00.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Notre jeu de données est enfin prêt pour effectuer diverses analyses.</div>", unsafe_allow_html=True)
    st.write("")
    st.write(all_players.head(10))
       
#%% Ajout du contenu de la page Visualisation 1ère approche
# Évaluation des joueurs de la saison dernière par ligue
def last_season_eval(df):
    col_2022_2023 = ['Joueur', 'Poste', 'Cote', 'Club', 'j38', 'j37', 'j36', 'j35', 'j34', 'j33',
                     'j32', 'j31', 'j30', 'j29','j28', 'j27', 'j26', 'j25', 'j24', 'j23',
                     'j22', 'j21', 'j20', 'j19','j18', 'j17', 'j16', 'j15', 'j14', 'j13',]

    df_2022_2023 = df[col_2022_2023]

    last_season = ['j38', 'j37', 'j36', 'j35', 'j34', 'j33', 'j32', 'j31', 'j30', 'j29',
                   'j28', 'j27', 'j26', 'j25', 'j24', 'j23', 'j22', 'j21', 'j20', 'j19',
                   'j18', 'j17', 'j16', 'j15', 'j14', 'j13']

    df_2022_2023["Note"] = round(df_2022_2023[last_season].sum(axis=1) / (df_2022_2023[last_season] != 0).sum(axis=1),1)

    df_2022_2023 = df_2022_2023.dropna(subset = ["Note"])
    df_2022_2023 = df_2022_2023.drop(last_season, axis = 1)
    df_2022_2023 = df_2022_2023.sort_values(["Note" , "Cote"], ascending = False) 
    df_2022_2023 = df_2022_2023.reset_index(drop = True)                                            
  
    return df_2022_2023

Ligue1_2022_2023 = last_season_eval(Ligue1_J9_raw)
Ligue2_2022_2023 = last_season_eval(Ligue2_J11_raw)
PremierLeague_2022_2023 = last_season_eval(PremierLeague_J9_raw)
Liga_2022_2023 = last_season_eval(Liga_J10_raw)
SerieA_2022_2023 = last_season_eval(SerieA_J9_raw)

# Évaluation des joueurs de la saison actuelle par ligue
def current_season_eval(df):
    col_2023_2024 = ['Joueur', 'Poste', 'Cote', 'Club', 'j11', 'j10', 'j9', 'j8', 'j7', 'j6',
                     'j5', 'j4', 'j3', 'j2', 'j1']

    df_2023_2024 = df[col_2023_2024]

    current_season = ['j11', 'j10', 'j9', 'j8', 'j7', 'j6', 'j5', 'j4', 'j3', 'j2',
                   'j1']

    df_2023_2024["Note"] = round(df_2023_2024[current_season].sum(axis=1) / (df_2023_2024[current_season] != 0).sum(axis=1),1)

    df_2023_2024 = df_2023_2024.dropna(subset = ["Note"])
    df_2023_2024 = df_2023_2024.drop(current_season, axis = 1)
    df_2023_2024 = df_2023_2024.sort_values(["Note" , "Cote"], ascending = False)
    df_2023_2024 = df_2023_2024.reset_index(drop = True) 
    
    return df_2023_2024

Ligue1_2023_2024 = current_season_eval(Ligue1_J9_raw)
Ligue2_2023_2024 = current_season_eval(Ligue2_J11_raw)
PremierLeague_2023_2024 = current_season_eval(PremierLeague_J9_raw)
Liga_2023_2024 = current_season_eval(Liga_J10_raw)
SerieA_2023_2024 = current_season_eval(SerieA_J9_raw)

# Fonction pour récupérer le nom des dataframes dans une variable
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

# Visualisation des notes des joueurs selon leur cote
def viz_notes(df):
  
    plot = px.scatter(df, x = "Cote", y = "Note", color = "Poste",
                     title = get_df_name(df),
                     color_discrete_map = {"A" : "#1f77b4",
                                           "MO" : "#ff7f0e",
                                           "MD" : "#2ca02c",
                                           "DL" : "#d62728",
                                           "DC" : "#9467bd",
                                           "G" : "#8c564b"},
                     category_orders= {"Poste": ["G", "DC", "DL", "MD", "MO", "A"]},
                     hover_data = ["Joueur", "Club", "Note", "Cote", "Poste"])
    
    return plot

# Comparaison des performances des top joueurs sur les deux saisons
Ligue1_eval = Ligue1_2022_2023.merge(Ligue1_2023_2024, on = ["Joueur", "Poste", "Cote", "Club"])
Ligue2_eval = Ligue2_2022_2023.merge(Ligue2_2023_2024, on = ["Joueur", "Poste", "Cote", "Club"])
PremierLeague_eval = PremierLeague_2022_2023.merge(PremierLeague_2023_2024, on = ["Joueur", "Poste", "Cote", "Club"])
Liga_eval = Liga_2022_2023.merge(Liga_2023_2024, on = ["Joueur", "Poste", "Cote", "Club"])
SerieA_eval = SerieA_2022_2023.merge(SerieA_2023_2024, on = ["Joueur", "Poste", "Cote", "Club"])

total_eval = pd.concat([Ligue1_eval, Ligue2_eval, PremierLeague_eval, Liga_eval, SerieA_eval], axis = 0).reset_index(drop = True).sort_values(["Note_x", "Note_y"], ascending = False)

def top_joueurs(df):
    
    top_A = df[df["Poste"] == "A"].head(10)
    top_MO = df[df["Poste"] == "MO"].head(10)
    top_MD = df[df["Poste"] == "MD"].head(10)
    top_DL = df[df["Poste"] == "DL"].head(10)
    top_DC = df[df["Poste"] == "DC"].head(10)
    top_G = df[df["Poste"] == "G"].head(10)
    
    return top_A, top_MO, top_MD, top_DL, top_DC, top_G

top_A = top_joueurs(total_eval)[0]
top_MO = top_joueurs(total_eval)[1]
top_MD = top_joueurs(total_eval)[2]
top_DL = top_joueurs(total_eval)[3]
top_DC = top_joueurs(total_eval)[4]
top_G = top_joueurs(total_eval)[5]

# Visualisation des performances des top joueurs sur les deux saisons
Ligue1_2022_2023["Saison"] = "2022_2023"
Ligue2_2022_2023["Saison"] = "2022_2023"
PremierLeague_2022_2023["Saison"] = "2022_2023"
Liga_2022_2023["Saison"] = "2022_2023"
SerieA_2022_2023["Saison"] = "2022_2023"

Ligue1_2023_2024["Saison"] = "2023_2024"
Ligue2_2023_2024["Saison"] = "2023_2024"
PremierLeague_2023_2024["Saison"] = "2023_2024"
Liga_2023_2024["Saison"] = "2023_2024"
SerieA_2023_2024["Saison"] = "2023_2024"

total_eval_viz = pd.concat([Ligue1_2022_2023, Ligue2_2022_2023, PremierLeague_2022_2023, Liga_2022_2023, SerieA_2022_2023,
                           Ligue1_2023_2024, Ligue2_2023_2024, PremierLeague_2023_2024, Liga_2023_2024, SerieA_2023_2024], ignore_index=True)

top_A_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_A["Joueur"].values)].sort_values(["Joueur", "Saison"])
top_MO_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_MO["Joueur"].values)].sort_values(["Joueur", "Saison"])
top_MD_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_MD["Joueur"].values)].sort_values(["Joueur", "Saison"])
top_DL_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_DL["Joueur"].values)].sort_values(["Joueur", "Saison"])
top_DC_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_DC["Joueur"].values)].sort_values(["Joueur", "Saison"])
top_G_viz = total_eval_viz[total_eval_viz["Joueur"].isin(top_G["Joueur"].values)].sort_values(["Joueur", "Saison"])


# Création des DataFrames spécialisés

# Sélection des joueurs actifs
active_players = all_players.loc[all_players["Nb match"] != 0]

# Sélection des colonnes spécifiques aux joueurs de champs (fp pour field player)
all_fp = all_players.loc[all_players["Poste"] != "G",["Joueur", "Poste", "Cote", "Var cote", "% achat", "% achat tour 1", "Q2 Toutes tailles", 'Q3 Toutes tailles', 'Q2 à 6 joueurs', 'Q3 à 6 joueurs',
                                                              'Q2 à 8 joueurs', 'Q3 à 8 joueurs', 'Q2 à 10 joueurs', 'Q3 à 10 joueurs', 'Note', 'Note série', 'Note 1 an', 'Note M11', "Nb match", "Nb match série",
                                                              "Nb match 1 an", 'But', "%Titu", "Temps", "Cleansheet", 'But/Peno', 'But/Coup-franc', 'But/surface', 'Pass decis.', 'Occas° créée',
                                                              'Tirs', 'Tirs cadrés', 'Corner gagné', '%Passes', 'Ballons', 'Interceptions', 'Tacles', '%Duel', 'Fautes', 'Poss Def',
                                                              'Poss Mil', 'Centres', 'Centres ratés', 'Dégagements', "But concédé", 'Ballon perdu', 'Passe>Tir', 'Passe perfo', 'Dépossédé', "Nb matchs gagnés",
                                                              'Erreur>But', "Diff de buts", 'Grosse occas manquée', 'Balle non rattrapée', "Bonus moy", "Malus moy", "Club", "Ligue"]]

# Sélection des colonnes spécifiques aux joueurs de champs actifs (Nb match != 0)
fp = all_fp.loc[all_fp["Nb match"] != 0]

# Sélection des colonnes spécifiques aux gardiens de buts (gk pour goalkeeper)
all_gk = all_players_raw.loc[all_players_raw["Poste"] == "G", ["Joueur", "Poste", "Cote", "Var cote", "% achat", "% achat tour 1", "Q2 Toutes tailles", 'Q3 Toutes tailles', 'Q2 à 6 joueurs', 'Q3 à 6 joueurs',
                                                              'Q2 à 8 joueurs', 'Q3 à 8 joueurs', 'Q2 à 10 joueurs', 'Q3 à 10 joueurs', 'Note', 'Note série', 'Note 1 an', 'Note M11', "Nb match", "Nb match série",
                                                              "Nb match 1 an", "%Titu", "Temps", "Cleansheet", 'But évité', 'Action stoppée', "But concédé", 'Plonge&stop', "Nb matchs gagnés", 'Erreur>But',
                                                              "Diff de buts", "Bonus moy", "Malus moy", "Club", "Ligue"]]

# Sélection des colonnes spécifiques aux gardiens actifs (Nb match != 0)
gk = all_gk.loc[all_gk["Nb match"] != 0]

# Études des corrélations des variables numériques
corr_col_fp = ['Note', "Nb match", 'But', "%Titu", "Temps", "Cleansheet", "But/Peno", "But/Coup-franc", "But/surface", "Pass decis.",
               "Occas° créée", 'Tirs', 'Tirs cadrés', 'Corner gagné', '%Passes', 'Ballons', 'Interceptions', 'Tacles', '%Duel', 'Fautes',
               'Poss Def', 'Poss Mil', 'Centres', 'Centres ratés', 'Dégagements', "But concédé", 'Ballon perdu', 'Passe>Tir', 'Passe perfo', 'Dépossédé',
               "Nb matchs gagnés", "Diff de buts", 'Grosse occas manquée', 'Balle non rattrapée', "Bonus moy", "Malus moy"]

corr_col_gk = ['Note', "Nb match", "%Titu", "Temps", "Cleansheet", "But évité", "Action stoppée", "But concédé", "Plonge&stop",
               "Nb matchs gagnés", "Erreur>But", "Diff de buts", "Bonus moy", "Malus moy"]

corr_fp = fp[corr_col_fp].corr() 
corr_A = fp.loc[fp["Poste"] == "A", corr_col_fp].corr() 
corr_MD = fp.loc[fp["Poste"] == "MD", corr_col_fp].corr() 
corr_MO = fp.loc[fp["Poste"] == "MO", corr_col_fp].corr() 
corr_DL = fp.loc[fp["Poste"] == "DL", corr_col_fp].corr() 
corr_DC = fp.loc[fp["Poste"] == "DC", corr_col_fp].corr() 
corr_gk = gk[corr_col_gk].corr()

# Récupération des colonnes essentielles pour la détermination de la note selon le poste
note_A = corr_A.iloc[0,:]
note_A = note_A.reset_index()
note_A = note_A[note_A["Note"] > 0.5].sort_values("Note", ascending=False)
note_A = note_A.iloc[1:]
note_MD = corr_MD.iloc[0,:]
note_MD = note_MD.reset_index()
note_MD = note_MD[note_MD["Note"] > 0.4].sort_values("Note", ascending=False)
note_MD = note_MD.iloc[1:]
note_MO = corr_MO.iloc[0,:]
note_MO = note_MO.reset_index()
note_MO = note_MO[note_MO["Note"] > 0.5].sort_values("Note", ascending=False)
note_MO = note_MO.iloc[1:]
note_DL = corr_DL.iloc[0,:]
note_DL = note_DL.reset_index()
note_DL = note_DL[note_DL["Note"] > 0.4].sort_values("Note", ascending=False)
note_DL = note_DL.iloc[1:]
note_DC = corr_DC.iloc[0,:]
note_DC = note_DC.reset_index()
note_DC = note_DC[note_DC["Note"] > 0.4].sort_values("Note", ascending=False)
note_DC = note_DC.iloc[1:]
note_gk = corr_gk.iloc[0,:]
note_gk = note_gk.reset_index()
note_gk = note_gk[note_gk["Note"] > 0.4].sort_values("Note", ascending=False)
note_gk = note_gk.iloc[1:]

import statsmodels.api
result = statsmodels.formula.api.ols("Cote ~ Note", data = active_players).fit()
df_result = statsmodels.api.stats.anova_lm(result)
#%% Mise en page de la page Visualisation 1ère approche
if page == pages[2] :
    st.write("## Visualisation 1ère approche")
    st.markdown("<div style='text-align: justify;'>Les visualisations présentées ici ont été élaborées selon une première approche consistant à repérer les meilleurs joueurs de manière visible, selon leur note et leur valeur d'achat.</div>", unsafe_allow_html=True)
    st.markdown("*Pour ne pas surcharger la présentation, seulement les graphes concernant la ligue 1 seront montrés*.")
    st.write("")
    st.markdown("<div style='text-align: justify;'>En étudiant dans un premier temps la distribution des notes par équipe, on peut facilement cibler les équipes dont certains joueurs sortent du lot. Il sera d'autant plus intéressant d'aller chercher ces «outliers» dans des équipes réputées faibles, afin d'en obtenir le meilleur prix.</div>", unsafe_allow_html=True)
    st.write("")
    
    fig, ax = plt.subplots()
    sns.boxplot(data = active_players[active_players["Ligue"] == "Ligue 1"], x = "Club", y = "Note", ax = ax, hue = "Club")
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribution des notes par équipe de Ligue 1")
    st.pyplot(fig)
   
    fig, ax = plt.subplots()    
    sns.boxplot(data = active_players, x = "Poste", y = "Note", hue = "Poste")
    plt.title("Distribution des notes par poste")
    st.pyplot(fig)
    
    st.markdown("<div style='text-align: justify;'>D'une manière générale, en regroupant nos joueurs par poste, on se rend compte que les meilleures notes sont obtenues par les joueurs offensifs. Cette constatation nous laisse à penser que ce sont également les joueurs les plus chers.</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div style='text-align: justify;'>Nos données nous permettent de vérifier cela en comparant directement les notes des joueurs selon leur prix de base. En plus de cela, nous faisons une différence entre les saisons pour voir l'évolution des joueurs. </div>", unsafe_allow_html=True)
    
    plot = viz_notes(Ligue1_2022_2023)
    plot.update_yaxes(range=[2, 9], row=1, col=1)
    plot.update_xaxes(range=[0, 45], row=1, col=1)
    st.plotly_chart(plot, use_container_width=True)

    plot = viz_notes(Ligue1_2023_2024)
    plot.update_yaxes(range=[2, 9], row=1, col=1)
    plot.update_xaxes(range=[0, 45], row=1, col=1)
    st.plotly_chart(plot, use_container_width=True)
    st.markdown("*Ces graphes sont interractifs. Survolez un point pour plus de détails. Double-cliquez sur un poste de la légende pour montrer uniquement cette catégorie*.")
    
    
    st.markdown("<div style='text-align: justify;'>L'idée de la relation entre le prix et la note des joueurs à l'air d'être globalement respectée sur ces visualisations. Pour en avoir le coeur net, on effectue un test Anova.</div>", unsafe_allow_html=True)
    st.write("")
    st.dataframe(df_result.style.format('{:.2f}'))
    st.write("")
    st.markdown("<div style='text-align: justify;'>Pour terminer cette première approche, on s'intéresse un peu plus en détail sur les performances des top joueurs sur la saison actuelle et la saison passée.</div>", unsafe_allow_html=True)
    st.markdown("*Ces performances concernent les joueurs de toutes les ligues*.")
    st.write("")

    fig, ax = plt.subplots()
    sns.barplot(data = top_G_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs gardiens de 2022/2023")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(data = top_DC_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs défenseurs centraux de 2022/2023")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(data = top_DL_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs latéraux défensifs de 2022/2023")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(data = top_MD_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs milieux defensifs de 2022/2023")
    st.pyplot(fig)

    fig, ax = plt.subplots()    
    sns.barplot(data = top_MO_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs milieux offensifs de 2022/2023")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data = top_A_viz, x = 'Joueur', y = "Note", hue = "Saison")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,10)
    plt.title("Comparaison des notes de la saison 2023/2024 des meilleurs attaquants de 2022/2023")   
    st.pyplot(fig)

    st.markdown("<div style='text-align: justify;'>En conclusion de cette première approche, on offre à l'utilisateur un moyen visuel de détection des joueurs en vue de son mercato. L'interactivé du plotly lui permet facilement d'identifier les joueurs correspondants à ses besoins.</div>", unsafe_allow_html=True)

#%% Ajout du contenu de la page Visualisation 2ème approche
# Études des corrélations des variables numériques
corr_col_fp = ['Note', "Nb match", 'But', "%Titu", "Temps", "Cleansheet", "But/Peno", "But/Coup-franc", "But/surface", "Pass decis.",
               "Occas° créée", 'Tirs', 'Tirs cadrés', 'Corner gagné', '%Passes', 'Ballons', 'Interceptions', 'Tacles', '%Duel', 'Fautes',
               'Poss Def', 'Poss Mil', 'Centres', 'Centres ratés', 'Dégagements', "But concédé", 'Ballon perdu', 'Passe>Tir', 'Passe perfo', 'Dépossédé',
               "Nb matchs gagnés", "Diff de buts", 'Grosse occas manquée', 'Balle non rattrapée', "Bonus moy", "Malus moy"]

corr_col_gk = ['Note', "Nb match", "%Titu", "Temps", "Cleansheet", "But évité", "Action stoppée", "But concédé", "Plonge&stop",
               "Nb matchs gagnés", "Erreur>But", "Diff de buts", "Bonus moy", "Malus moy"]

corr_fp = fp[corr_col_fp].corr()
corr_A = fp.loc[fp["Poste"] == "A", corr_col_fp].corr()
corr_MD = fp.loc[fp["Poste"] == "MD", corr_col_fp].corr()
corr_MO = fp.loc[fp["Poste"] == "MO", corr_col_fp].corr()
corr_DL = fp.loc[fp["Poste"] == "DL", corr_col_fp].corr()
corr_DC = fp.loc[fp["Poste"] == "DC", corr_col_fp].corr()
corr_gk = gk[corr_col_gk].corr()

# Récupération des colonnes essentielles pour la détermination de la note selon le poste
note_A = corr_A.iloc[0,:]
note_A = note_A.reset_index()
note_A = note_A[note_A["Note"] > 0.5].sort_values("Note", ascending=False)
note_A = note_A.iloc[1:]
note_MD = corr_MD.iloc[0,:]
note_MD = note_MD.reset_index()
note_MD = note_MD[note_MD["Note"] > 0.4].sort_values("Note", ascending=False)
note_MD = note_MD.iloc[1:]
note_MO = corr_MO.iloc[0,:]
note_MO = note_MO.reset_index()
note_MO = note_MO[note_MO["Note"] > 0.5].sort_values("Note", ascending=False)
note_MO = note_MO.iloc[1:]
note_DL = corr_DL.iloc[0,:]
note_DL = note_DL.reset_index()
note_DL = note_DL[note_DL["Note"] > 0.4].sort_values("Note", ascending=False)
note_DL = note_DL.iloc[1:]
note_DC = corr_DC.iloc[0,:]
note_DC = note_DC.reset_index()
note_DC = note_DC[note_DC["Note"] > 0.4].sort_values("Note", ascending=False)
note_DC = note_DC.iloc[1:]
note_gk = corr_gk.iloc[0,:]
note_gk = note_gk.reset_index()
note_gk = note_gk[note_gk["Note"] > 0.4].sort_values("Note", ascending=False)
note_gk = note_gk.iloc[1:]

#%% Mise en page de la page Visualisation 1ère approche
if page == pages[3] :
    st.write("## Visualisation 2ème approche")
    st.markdown("<div style='text-align: justify;'>La deuxième approche de notre étude consiste à aller plus loin dans l'évaluation des joueurs en essayant de prédire leurs performances futures. Pour ce faire, nous avons mener une enquête préalable sur la corrélation entre chaque statistique afin de choisir les colonnes les plus importantes pour que notre modèle de machine learning puisse déterminer les notes des joueurs.</div>", unsafe_allow_html=True)
    st.markdown("*Dans un souci de visibilité, les cartes thermiques (ou heatmap) ne seront pas affichées*.")

    fig, ax = plt.subplots()
    sns.barplot(data = note_gk, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les gardiens")    
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data = note_DC, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les défenseurs centraux")    
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data = note_DL, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les défenseurs latéraux")    
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data = note_MD, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les milieux défensifs")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data = note_MO, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les milieux offensifs")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(data = note_A, x = "index", y = "Note", color="b")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("Importance")
    plt.title("Importance des colonnes dans la détermination de la note pour les attaquants")    
    st.pyplot(fig)

#%% Ajout du contenu nécessaire à la régression linéaire
col_ML = ['Poste', 'Nb match', 'But', '%Titu', 'Temps', 'Cleansheet', 'Pass decis.', 'Occas° créée', 'Tirs', 'Tirs cadrés',
              'Corner gagné', 'Ballons', 'Tacles', 'But évité', 'Action stoppée', 'Poss Def', 'Poss Mil', 'Ballon perdu', 'Passe>Tir', 'Passe perfo',
              'Plonge&stop', 'Nb matchs gagnés', 'Diff de buts', 'Bonus moy', 'Malus moy']

col_cible = ["Note"]

data_rl = all_players[col_ML]
target_rl = all_players[col_cible]
X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(data_rl, target_rl, test_size = 0.25, random_state = 27)
X_train_rl = X_train_rl.fillna(0)
X_test_rl = X_test_rl.fillna(0)
X_train_rl = pd.get_dummies(X_train_rl, columns = ["Poste"], dtype = 'int')
X_test_rl = pd.get_dummies(X_test_rl, columns = ["Poste"], dtype = 'int')
scaler = StandardScaler()
col_scaler = ['Nb match', 'But', '%Titu', 'Temps', 'Cleansheet', 'Pass decis.', 'Occas° créée', 'Tirs', 'Tirs cadrés', 'Corner gagné',
              'Ballons', 'Tacles', 'But évité', 'Action stoppée', 'Poss Def', 'Poss Mil', 'Ballon perdu', 'Passe>Tir', 'Passe perfo', 'Plonge&stop',
              'Nb matchs gagnés', 'Diff de buts', 'Bonus moy', 'Malus moy']
X_train_rl[col_scaler] = scaler.fit_transform(X_train_rl[col_scaler])
X_test_rl[col_scaler] = scaler.transform(X_test_rl[col_scaler])
model_rl = LinearRegression()
model_rl.fit(X_train_rl, y_train_rl)
model_rl.score(X_train_rl, y_train_rl)
model_rl.score(X_test_rl, y_test_rl)
y_pred_rl = model_rl.predict(X_test_rl)

#%% Mise en page de la page Modélisation : Régression Linéaire
if page == pages[4] :
    st.write("## Modélisation : Régression Linéaire")
    st.markdown("<div style='text-align: justify;'>Notre variable cible étant la note des joueurs, soit une variable quantitative, nous avons dans un premier temps testé un modèle de régression linéaire.</div>", unsafe_allow_html=True)

    st.write(model_rl.score(X_test_rl, y_test_rl))
    st.markdown("<div style='text-align: justify;'>Nous obtenons un score plutôt intéressant que nous pouvons représenter sous la forme d'une ligne de régression.</div>", unsafe_allow_html=True)
    st.write("")
    
    fig, ax = plt.subplots()
    plt.scatter(y_pred_rl, y_test_rl)
    plt.plot((y_test_rl.min(), y_test_rl.max()), (y_test_rl.min(), y_test_rl.max()), color = 'red')
    plt.xlabel("prediction")
    plt.ylabel("vrai valeur")
    plt.title('Régression Linéaire pour la prédiction des notes des joueurs')    
    st.pyplot(fig)  
    
    st.markdown("*Le score variera à chaque changement de page car l'algorithme s'entraîne à chaque fois sur des données tirées au hasard*.")

#%% Ajout du contenu nécessaire à la première classification
q1, q3, q5, q7, q9 = active_players["Note"].quantile(q = [0.1, 0.3, 0.5, 0.7, 0.9])

col_class = ['Poste', 'Nb match', 'But', '%Titu', 'Temps', 'Cleansheet', 'Pass decis.', 'Occas° créée', 'Tirs', 'Tirs cadrés',
             'Corner gagné', 'Ballons', 'Tacles', 'But évité', 'Action stoppée', 'Poss Def', 'Poss Mil', 'Ballon perdu', 'Passe>Tir', 'Passe perfo',
             'Plonge&stop', 'Nb matchs gagnés', 'Diff de buts', 'Bonus moy', 'Malus moy', 'Note']

data_class = all_players[col_class]
data_class["eval"] = pd.cut(data_class["Note"], [-np.inf, 0, q1, q3, q5, q7, q9, 10], labels=[0, 1, 2, 3, 4, 5, 6])
target_class = data_class["eval"]
data_class = data_class[col_ML]
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(data_class, target_class, test_size = 0.25, random_state = 27)

X_train_class = X_train_class.fillna(0)
X_test_class = X_test_class.fillna(0)

X_train_class = pd.get_dummies(X_train_class, columns = ["Poste"], dtype = 'int')
X_test_class = pd.get_dummies(X_test_class, columns = ["Poste"], dtype = 'int')

X_train_class[col_scaler] = scaler.fit_transform(X_train_class[col_scaler])
X_test_class[col_scaler] = scaler.transform(X_test_class[col_scaler])

def prediction(classifier):
   if classifier == 'Régression Logistique':
     clf = LogisticRegression()
   elif classifier == 'Arbre de Décission':
     clf = DecisionTreeClassifier()    
   elif classifier == 'Forêt Aléatoire':
     clf = RandomForestClassifier()    
   clf.fit(X_train_class, y_train_class)
   return clf

choix = ['Régression Logistique', 'Arbre de Décission', 'Forêt Aléatoire']

def scores(clf, choice):
   if choice == 'Score':
     return clf.score(X_test_class, y_test_class)
   elif choice == 'Matrice de Confusion':
     return confusion_matrix(y_test_class, clf.predict(X_test_class))
   elif choice == 'Rapport de Classification':
     target_names = ["classe 0", "classe 1", "classe 2", "classe 3", "classe 4", "classe 5", "classe 6"]
     y_pred_class = clf.predict(X_test_class)
     return pd.DataFrame(classification_report(y_test_class, y_pred_class, target_names=target_names, output_dict=True))

       
#%% Mise en page de la page Modélisation : Classification - 1ère partie
if page == pages[5] :
    st.write("## Modélisation : Classification - 1ère partie")
    st.markdown("<div style='text-align: justify;'>Notre variable cible <b>note</b> pouvant également être considérée comme une variable qualitative, nous pouvons également appliquer les modèles de classification suivants :</div>", unsafe_allow_html=True)
    st.markdown("- la régression logistique")
    st.markdown("- l'arbre de décission")
    st.markdown("- la forêt aléatoire")    
    st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Dans un premier temps, nous décomposons notre variable cible en sept classes différentes selon les quantiles 1-3-5-7-9 de la <b>note</b> :</div>", unsafe_allow_html=True)
    st.markdown("- classe 0 : pas de matchs joués")
    st.markdown("- classe 1 : mauvais")
    st.markdown("- classe 2 : pas bon")
    st.markdown("- classe 3 : assez bon")
    st.markdown("- classe 4 : bon")
    st.markdown("- classe 5 : très bon")
    st.markdown("- classe 6 : exceptionnel")
    st.markdown("<div style='text-align: justify;'>Nous obtenons ainsi les résultats suivants.</div>", unsafe_allow_html=True)

    option = st.selectbox('Choix du modèle', choix)
    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Score', 'Matrice de Confusion', 'Rapport de Classification'))
    if display == 'Score':
        st.write(scores(clf, display))
    elif display == 'Matrice de Confusion':
        st.dataframe(scores(clf, display))
    elif display == 'Rapport de Classification':
        st.dataframe(scores(clf, display).transpose())
    st.markdown("*Le score variera à chaque changement d'options car l'algorithme s'entraîne à chaque fois sur des données tirées au hasard*.")
       
    st.markdown("<div style='text-align: justify;'>On constate que les résultats obtenus sont moins performants que notre régression linéaire.</div>", unsafe_allow_html=True)
        
#%% Ajout du contenu nécessaire à la deuxième classification
data = all_players[col_class]
data["eval"] = pd.cut(data["Note"], [-np.inf, 0, q1, q7, q9, 10], labels=[0, 1, 3, 5, 6])
target = data["eval"]
data = data[col_ML]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state = 27)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
X_train = pd.get_dummies(X_train, columns = ["Poste"], dtype = 'int')
X_test = pd.get_dummies(X_test, columns = ["Poste"], dtype = 'int')
X_train[col_scaler] = scaler.fit_transform(X_train[col_scaler])
X_test[col_scaler] = scaler.transform(X_test[col_scaler])

new_choix = ['Régression Logistique', 'Arbre de Décission']

def new_prediction(classifier):
   if classifier == 'Régression Logistique':
     clf = LogisticRegression(random_state = 27)
   elif classifier == 'Arbre de Décission':
     clf = DecisionTreeClassifier(random_state = 27)    
   elif classifier == 'Forêt Aléatoire':
     clf = RandomForestClassifier(random_state = 27)    
   clf.fit(X_train, y_train)
   return clf

def new_scores(clf, choice):
   if choice == 'Score':
       return clf.score(X_test, y_test)
   elif choice == 'Matrice de Confusion':
       return confusion_matrix(y_test, clf.predict(X_test))
   elif choice == 'Rapport de Classification':
       target_names = ["classe 0", "classe 1", "classe 3", "classe 5", "classe 6"]
       y_pred = clf.predict(X_test)
       return pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
       
#%% Mise en page de la page Modélisation : Classification - 2ème partie
if page == pages[6] : 
  st.write("## Modélisation : Classification - 2ème partie")    
  st.markdown("<div style='text-align: justify;'>Pour obtenir des résultats plus probant avec nos algorithmes de classification, on se résoud à réduire le nombre de classe de la manière suivante :</div>", unsafe_allow_html=True)
  st.markdown("- classe 0 : pas de matchs joués")
  st.markdown("- classe 1 : mauvais")
  st.markdown("- classe 3 : bon")
  st.markdown("- classe 5 : très bon")
  st.markdown("- classe 6 : exceptionnel")
  st.markdown("<div style='text-align: justify;'>Au vue des résultats peu probant de la forêt aléatoire, on décide également de l'exclure de nos modèles.</div>", unsafe_allow_html=True)

  option = st.selectbox('Choix du modèle', new_choix)
  clf = new_prediction(option)
  display = st.radio('Que souhaitez-vous montrer ?', ('Score', 'Matrice de Confusion', 'Rapport de Classification'))
  if display == 'Score':
      st.write(new_scores(clf, display))
  elif display == 'Matrice de Confusion':
      st.dataframe(new_scores(clf, display))
  elif display == 'Rapport de Classification':
      st.dataframe(new_scores(clf, display).transpose())
      
  st.markdown("<div style='text-align: justify;'>On remarque tout de suite une amélioration des résultats pour les deux modèles. On peut dès lors les utiliser pour créer nos outils de sélection de joueur.</div>", unsafe_allow_html=True)
  # data = pd.get_dummies(data_rl, columns = ["Poste"], dtype = 'int')
  # data = data.fillna(0)
  # clf.fit(data, target)
  # st.write(clf.predict(data))
#%% Ajout du contenu nécessaire à la fonction Mercato
data = data.fillna(0)
data = pd.get_dummies(data, columns = ["Poste"], dtype = 'int')

def mercato(liste_joueurs, ligue, budget, tactique, model):
    model.fit(data, target)
    prediction = model.predict(data)
    # Création de notre liste de joueurs
    players_mercato = liste_joueurs
    prediction = pd.Series(prediction, name = "prediction")
    liste_joueurs = pd.concat([liste_joueurs, prediction], axis = 1)
    players_mercato = liste_joueurs[["Joueur", "Poste", "Cote", "Q2 Toutes tailles", "Q3 Toutes tailles", "Note", "Nb match", "Club", "Ligue", "prediction"]]

    # On filtre pour récupérer notre liste de joueur selon la ligue
    if ligue == "Toutes":
        players_mercato = players_mercato
    elif ligue == "Ligue 1":
        players_mercato = players_mercato[players_mercato["Ligue"] == "Ligue 1"]
    elif ligue == "Ligue 2":
        players_mercato = players_mercato[players_mercato["Ligue"] == "Ligue 2"]
    elif ligue == "Liga":
        players_mercato = players_mercato[players_mercato["Ligue"] == "Liga"]
    elif ligue == "Premier League":
        players_mercato = players_mercato[players_mercato["Ligue"] == "Premier League"]   
    elif ligue == "Serie A":
        players_mercato = players_mercato[players_mercato["Ligue"] == "Serie A"]
    # Sélection des meilleurs joueurs ayant joué au moins 75% des matchs
    A = players_mercato.loc[(players_mercato["Poste"] == "A") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    A = A.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False])
    MO = players_mercato.loc[(players_mercato["Poste"] == "MO") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    MO = MO.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False])
    MD = players_mercato.loc[(players_mercato["Poste"] == "MD") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    MD = MD.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False])
    DL = players_mercato.loc[(players_mercato["Poste"] == "DL") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    DL = DL.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False])
    DC = players_mercato.loc[(players_mercato["Poste"] == "DC") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    DC = DC.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False])
    G = players_mercato.loc[(players_mercato["Poste"] == "G") & (players_mercato["Nb match"] >= 0.75 * players_mercato["Nb match"].max())]
    G = G.sort_values(["prediction", "Q3 Toutes tailles", "Note"], ascending = [False, True, False]) 
    # Sélection des joueurs selon les tactiques
    joueurs = pd.DataFrame()
    if tactique == "4-4-2":
      joueurs = pd.concat([joueurs, A.iloc[:4]])
      joueurs = pd.concat([joueurs, MO.iloc[:4]])
      joueurs = pd.concat([joueurs, MD.iloc[:4]])
      joueurs = pd.concat([joueurs, DL.iloc[:4]])
      joueurs = pd.concat([joueurs, DC.iloc[:4]])
      joueurs = pd.concat([joueurs, G.iloc[:2]])
    elif tactique == "4-2-1-3":
      joueurs = pd.concat([joueurs, A.iloc[:6]])
      joueurs = pd.concat([joueurs, MO.iloc[:2]])   
      joueurs = pd.concat([joueurs, MD.iloc[:4]])         
      joueurs = pd.concat([joueurs, DL.iloc[:4]])      
      joueurs = pd.concat([joueurs, DC.iloc[:4]])
      joueurs = pd.concat([joueurs, G.iloc[:2]])
    elif tactique == "4-1-2-3":
      joueurs = pd.concat([joueurs, A.iloc[:6]])
      joueurs = pd.concat([joueurs, MO.iloc[:4]])   
      joueurs = pd.concat([joueurs, MD.iloc[:2]])         
      joueurs = pd.concat([joueurs, DL.iloc[:4]])      
      joueurs = pd.concat([joueurs, DC.iloc[:4]])
      joueurs = pd.concat([joueurs, G.iloc[:2]]) 
    elif tactique == "4-3-3":
      joueurs = pd.concat([joueurs, A.iloc[:6]])
      joueurs = pd.concat([joueurs, MO.iloc[:3]])
      joueurs = pd.concat([joueurs, MD.iloc[:3]])
      joueurs = pd.concat([joueurs, DL.iloc[:4]])
      joueurs = pd.concat([joueurs, DC.iloc[:4]])
      joueurs = pd.concat([joueurs, G.iloc[:2]])
    elif tactique == "3-5-2":
      joueurs = pd.concat([joueurs, A.iloc[:4]])
      joueurs = pd.concat([joueurs, MO.iloc[:5]])
      joueurs = pd.concat([joueurs, MD.iloc[:3]])
      joueurs = pd.concat([joueurs, DL.iloc[:4]])
      joueurs = pd.concat([joueurs, DC.iloc[:4]])
      joueurs = pd.concat([joueurs, G.iloc[:2]])
    elif tactique == "5-3-2":
      joueurs = pd.concat([joueurs, A.iloc[:4]])
      joueurs = pd.concat([joueurs, MO.iloc[:3]])
      joueurs = pd.concat([joueurs, MD.iloc[:4]])
      joueurs = pd.concat([joueurs, DL.iloc[:4]])
      joueurs = pd.concat([joueurs, DC.iloc[:5]])
      joueurs = pd.concat([joueurs, G.iloc[:2]])
    # Vérification du budget
    budget = budget - sum(joueurs["Q3 Toutes tailles"])
    # Sélection de top players avec le budget restant
    top_players = players_mercato[~players_mercato.isin(joueurs.to_dict(orient = "list")).all(axis= 1)]
    top_players_budget = top_players[(top_players["Q3 Toutes tailles"] <= budget) & (top_players["Nb match"] >= 0.75 * top_players["Nb match"].max())]
    top_players_budget = top_players_budget.sort_values(["prediction", "Note"], ascending = [False, False]).head(20)
    # Sélection des top players restant (hors budget)
    top_players_restant = top_players[top_players["Nb match"] >= 0.75 * top_players["Nb match"].max()]
    top_players_restant = top_players_restant.sort_values(["prediction", "Note"], ascending = [False, False]).head(20)
    # Récupération des statistiques des joueurs
    stats_col = ["Joueur", "Poste", "Club", "Ligue", "Cote", "Q2 Toutes tailles", 'Q3 Toutes tailles', 'Note', 'Note série', 'Note 1 an',
                  'Note M11', "Nb match", "Nb match série", "Nb match 1 an", 'But', "%Titu", "Temps", "Cleansheet",'But/Peno', 'But/Coup-franc',
                  'But/surface', 'Pass decis.', 'Occas° créée', 'Tirs', 'Tirs cadrés', 'Corner gagné', '%Passes', 'Ballons', 'Interceptions', 'Tacles',
                  '%Duel', 'Fautes', 'But évité', 'Action stoppée', 'Poss Def', 'Poss Mil', 'Centres', 'Centres ratés', 'Dégagements', "But concédé",
                  'Ballon perdu', 'Passe>Tir', 'Passe perfo', 'Dépossédé', 'Plonge&stop', "Nb matchs gagnés", 'Erreur>But', "Diff de buts", 'Grosse occas manquée', 'Balle non rattrapée',
                  "Bonus moy", "Malus moy", "prediction"]  
    index_commun = liste_joueurs.index.intersection(joueurs.index)
    joueurs = liste_joueurs[liste_joueurs.index.isin(index_commun)]
    joueurs = joueurs[stats_col]
    ordre = {"G" : 1, "DC" : 2, "DL" : 3, "MD" : 4, "MO" : 5, "A" : 6}
    joueurs = joueurs.sort_values(by = "Poste", key = lambda x: x.map(ordre)).reset_index(drop = True)   

    return (joueurs, budget, top_players_budget, top_players_restant)
     
#%% Mise en page de la page Fonction : Mercato
if page == pages[7] : 
  st.write("## Fonction : Mercato")
  st.markdown("<div style='text-align: justify;'>L'objectif de notre fonction mercato est de constituer la meilleure équipe au meilleur prix. Elle se compose en trois étapes :</div>", unsafe_allow_html=True)
  st.markdown("- Suggestion d'un effectif")
  st.markdown("- Évaluation du budget restant")
  st.markdown("- Suggestion de joueurs supplémentaires")
  st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)
  st.markdown("<div style='text-align: justify;'>Notre première étape consiste à confier à notre modèle une liste de joueurs afin qu'il détermine quels joueurs vont performer dans les prochains matchs. Cette liste est ensuite triée selon la valeure d'enchère maximale du joueur pour récupérer ceux dont le prix est bon marché. À noter que l'on s'intéresse uniquement aux joueurs ayant joué au moins 75% des matchs.</div>", unsafe_allow_html=True)
  st.markdown("<div style='text-align: justify;'>Pour ce faire, notre fonction mercato prend cinq arguements :", unsafe_allow_html=True)
  st.markdown("- Une liste globale de joueur")
  st.markdown("- Une ligue en particulier")
  st.markdown("- Un budget")
  st.markdown("- Une tactique")
  st.markdown("- Un modèle de classification")

  option = st.selectbox('Choix du modèle', new_choix)
  clf = new_prediction(option)
  
  col1, col2, col3, col4, col5 = st.columns(5)
  choix1 = ["Tous les joueurs"]
  choix2 = ["Toutes", "Ligue 1", "Ligue 2", "Liga", "Premier League", "Serie A"]
  choix3 = [500]
  choix4 = ["4-4-2", "4-3-3", "4-2-1-3", "4-1-2-3", "3-5-2", "5-3-2"]
  choix5 = ["Régression Logistique", "Arbre de Décission"]
 
  with col1:
      option1 = st.selectbox('Liste de joueur', choix1)
  with col2:
      option2 = st.selectbox('Ligue', choix2)      
  with col3:
      option3 = st.selectbox('Budget', choix3)
  with col4:
      option4 = st.selectbox('Tactique', choix4)
      
  if option1 == "Tous les joueurs":
    liste_joueurs = all_players

  ligue = option2
  budget = option3
  tactique = option4
  st.write("Suggestion d'effectif")
  st.dataframe(mercato(liste_joueurs, ligue, budget, tactique, clf)[0])
  st.write("Budget restant", mercato(liste_joueurs, ligue, budget, tactique, clf)[1])
  if mercato(liste_joueurs, ligue, budget, tactique, clf)[1] < 0:
      st.write("Réduire les enchères ou le nombre de joueurs")
  st.markdown("*Le modèle est calculé sur le 3ème quartile d'achat d'enchère des joueurs, ce qui signifie que les joueurs peuvent être disponibles pour des enchères moins élevées. Se référer à la colonne Q2 pour voir la valeur moyenne des enchères*.")
  st.write("Suggestion de joueurs supplémentaires pour le budget restant")
  st.dataframe(mercato(liste_joueurs, ligue, budget, tactique, clf)[2])
  st.write("Suggestion des tops joueurs restants")
  st.dataframe(mercato(liste_joueurs, ligue, budget, tactique, clf)[3])
 

#%% Ajout du contenu nécessaire à la fonction Composition
ligue1_442_LR = mercato(all_players, "Ligue 1", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
ligue1_433_LR = mercato(all_players, "Ligue 1", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
ligue1_4123_LR = mercato(all_players, "Ligue 1", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
ligue1_4213_LR = mercato(all_players, "Ligue 1", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
ligue1_352_LR = mercato(all_players, "Ligue 1", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
ligue1_532_LR = mercato(all_players, "Ligue 1", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

ligue1_442_DTC = mercato(all_players, "Ligue 1", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
ligue1_433_DTC = mercato(all_players, "Ligue 1", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
ligue1_4123_DTC = mercato(all_players, "Ligue 1", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
ligue1_4213_DTC = mercato(all_players, "Ligue 1", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
ligue1_352_DTC = mercato(all_players, "Ligue 1", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
ligue1_532_DTC = mercato(all_players, "Ligue 1", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

ligue2_442_LR = mercato(all_players, "Ligue 2", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
ligue2_433_LR = mercato(all_players, "Ligue 2", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
ligue2_4123_LR = mercato(all_players, "Ligue 2", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
ligue2_4213_LR = mercato(all_players, "Ligue 2", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
ligue2_352_LR = mercato(all_players, "Ligue 2", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
ligue2_532_LR = mercato(all_players, "Ligue 2", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

ligue2_442_DTC = mercato(all_players, "Ligue 2", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
ligue2_433_DTC = mercato(all_players, "Ligue 2", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
ligue2_4123_DTC = mercato(all_players, "Ligue 2", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
ligue2_4213_DTC = mercato(all_players, "Ligue 2", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
ligue2_352_DTC = mercato(all_players, "Ligue 2", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
ligue2_532_DTC = mercato(all_players, "Ligue 2", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

premierleague_442_LR = mercato(all_players, "Premier League", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
premierleague_433_LR = mercato(all_players, "Premier League", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
premierleague_4123_LR = mercato(all_players, "Premier League", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
premierleague_4213_LR = mercato(all_players, "Premier League", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
premierleague_352_LR = mercato(all_players, "Premier League", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
premierleague_532_LR = mercato(all_players, "Premier League", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

premierleague_442_DTC = mercato(all_players, "Premier League", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
premierleague_433_DTC = mercato(all_players, "Premier League", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
premierleague_4123_DTC = mercato(all_players, "Premier League", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
premierleague_4213_DTC = mercato(all_players, "Premier League", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
premierleague_352_DTC = mercato(all_players, "Premier League", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
premierleague_532_DTC = mercato(all_players, "Premier League", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

liga_442_LR = mercato(all_players, "Liga", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
liga_433_LR = mercato(all_players, "Liga", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
liga_4123_LR = mercato(all_players, "Liga", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
liga_4213_LR = mercato(all_players, "Liga", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
liga_352_LR = mercato(all_players, "Liga", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
liga_532_LR = mercato(all_players, "Liga", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

liga_442_DTC = mercato(all_players, "Liga", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
liga_433_DTC = mercato(all_players, "Liga", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
liga_4123_DTC = mercato(all_players, "Liga", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
liga_4213_DTC = mercato(all_players, "Liga", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
liga_352_DTC = mercato(all_players, "Liga", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
liga_532_DTC = mercato(all_players, "Liga", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

serieA_442_LR = mercato(all_players, "Serie A", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
serieA_433_LR = mercato(all_players, "Serie A", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
serieA_4123_LR = mercato(all_players, "Serie A", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
serieA_4213_LR = mercato(all_players, "Serie A", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
serieA_352_LR = mercato(all_players, "Serie A", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
serieA_532_LR = mercato(all_players, "Serie A", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

serieA_442_DTC = mercato(all_players, "Serie A", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
serieA_433_DTC = mercato(all_players, "Serie A", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
serieA_4123_DTC = mercato(all_players, "Serie A", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
serieA_4213_DTC = mercato(all_players, "Serie A", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
serieA_352_DTC = mercato(all_players, "Serie A", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
serieA_532_DTC = mercato(all_players, "Serie A", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

all_442_LR = mercato(all_players, "Tous les joueurs", 500, "4-4-2", LogisticRegression(random_state = 27))[0]
all_433_LR = mercato(all_players, "Tous les joueurs", 500, "4-3-3", LogisticRegression(random_state = 27))[0]
all_4123_LR = mercato(all_players, "Tous les joueurs", 500, "4-1-2-3", LogisticRegression(random_state = 27))[0]
all_4213_LR = mercato(all_players, "Tous les joueurs", 500, "4-2-1-3", LogisticRegression(random_state = 27))[0]
all_352_LR = mercato(all_players, "Tous les joueurs", 500, "3-5-2", LogisticRegression(random_state = 27))[0]
all_532_LR = mercato(all_players, "Tous les joueurs", 500, "5-3-2", LogisticRegression(random_state = 27))[0]

all_442_DTC = mercato(all_players, "Tous les joueurs", 500, "4-4-2", DecisionTreeClassifier(random_state = 27))[0]
all_433_DTC = mercato(all_players, "Tous les joueurs", 500, "4-3-3", DecisionTreeClassifier(random_state = 27))[0]
all_4123_DTC = mercato(all_players, "Tous les joueurs", 500, "4-1-2-3", DecisionTreeClassifier(random_state = 27))[0]
all_4213_DTC = mercato(all_players, "Tous les joueurs", 500, "4-2-1-3", DecisionTreeClassifier(random_state = 27))[0]
all_352_DTC = mercato(all_players, "Tous les joueurs", 500, "3-5-2", DecisionTreeClassifier(random_state = 27))[0]
all_532_DTC = mercato(all_players, "Tous les joueurs", 500, "5-3-2", DecisionTreeClassifier(random_state = 27))[0]

def composition(effectif, tactique):
    
    if tactique == "4-4-2":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(2).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(2).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(2).tolist())
        
        defenseur_lateral = []
        defenseur_laterals = effectif[effectif["Poste"] == "DL"].sort_values("Note série", ascending = False)
        defenseur_lateral.extend(defenseur_laterals["Joueur"].head(2).tolist())
        
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(2).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_lateral" : defenseur_lateral, "defenseur_central" : defenseur_central, "gardien" : gardien})

    elif tactique == "4-3-3":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(3).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(2).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(1).tolist())
        
        defenseur_lateral = []
        defenseur_laterals = effectif[effectif["Poste"] == "DL"].sort_values("Note série", ascending = False)
        defenseur_lateral.extend(defenseur_laterals["Joueur"].head(2).tolist())
        
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(2).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_lateral" : defenseur_lateral, "defenseur_central" : defenseur_central, "gardien" : gardien})
        
    elif tactique == "4-1-2-3":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(3).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(2).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(1).tolist())
        
        defenseur_lateral = []
        defenseur_laterals = effectif[effectif["Poste"] == "DL"].sort_values("Note série", ascending = False)
        defenseur_lateral.extend(defenseur_laterals["Joueur"].head(2).tolist())
        
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(2).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_lateral" : defenseur_lateral, "defenseur_central" : defenseur_central, "gardien" : gardien})

    elif tactique == "4-2-1-3":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(3).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(1).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(2).tolist())
        
        defenseur_lateral = []
        defenseur_laterals = effectif[effectif["Poste"] == "DL"].sort_values("Note série", ascending = False)
        defenseur_lateral.extend(defenseur_laterals["Joueur"].head(2).tolist())
        
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(2).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_lateral" : defenseur_lateral, "defenseur_central" : defenseur_central, "gardien" : gardien})

    elif tactique == "3-5-2":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(2).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(3).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(2).tolist())       
       
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(3).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_central" : defenseur_central, "gardien" : gardien})

    elif tactique == "5-3-2":
        attaque = []
        attaquants = effectif[effectif["Poste"] == "A"].sort_values("Note série", ascending = False)
        attaque.extend(attaquants["Joueur"].head(2).tolist())
        
        milieu_off = []
        milieux_off = effectif[effectif["Poste"] == "MO"].sort_values("Note série", ascending = False)
        milieu_off.extend(milieux_off["Joueur"].head(1).tolist())
        
        milieu_def = []
        milieux_def = effectif[effectif["Poste"] == "MD"].sort_values("Note série", ascending = False)
        milieu_def.extend(milieux_def["Joueur"].head(2).tolist())
        
        defenseur_lateral = []
        defenseur_laterals = effectif[effectif["Poste"] == "DL"].sort_values("Note série", ascending = False)
        defenseur_lateral.extend(defenseur_laterals["Joueur"].head(2).tolist())
        
        defenseur_central = []
        defenseur_centraux = effectif[effectif["Poste"] == "DC"].sort_values("Note série", ascending = False)
        defenseur_central.extend(defenseur_centraux["Joueur"].head(3).tolist())    
        
        gardien = []
        gardiens = effectif[effectif["Poste"] == "G"].sort_values("Note série", ascending = False)
        gardien.extend(gardiens["Joueur"].head(1).tolist())
        
        titulaire = {}
        titulaire.update({"attaque" : attaque, "milieu_off": milieu_off, "milieu_def" : milieu_def, "defenseur_lateral" : defenseur_lateral, "defenseur_central" : defenseur_central, "gardien" : gardien})
       
    return (titulaire)

# Visualisation compo
def compo_viz(compo, tactique):
    
    # Création de la figure
    fig=plt.figure(figsize= (15,15))
    ax=fig.add_subplot(1,1,1)

    # Création du terrain
    pitch_dark = matplotlib.patches.Rectangle((0, 0), 60, 90, color= '#adeb8a')
    pitch_light1 = matplotlib.patches.Rectangle((0, 10), 60, 10, color= '#4cd406')
    pitch_light2 = matplotlib.patches.Rectangle((0, 30), 60, 10, color= '#4cd406')
    pitch_light3 = matplotlib.patches.Rectangle((0, 50), 60, 10, color= '#4cd406')
    pitch_light4 = matplotlib.patches.Rectangle((0, 70), 60, 10, color= '#4cd406')
    
    ax.add_patch(pitch_dark)
    ax.add_patch(pitch_light1)
    ax.add_patch(pitch_light2)
    ax.add_patch(pitch_light3)
    ax.add_patch(pitch_light4)
    
    plt.plot([0,0],[0,90], color="black")
    plt.plot([60,60],[0,90], color="black")
    plt.plot([0,60],[0,0], color="black")
    plt.plot([0,60],[90,90], color="black")
    
    plt.ylim([0,90])
    plt.xlim([0,60])
   
    # Création de la surface de réparation
    plt.plot([18,42],[18,18],color="black")
    plt.plot([18,18],[0,18],color="black")
    plt.plot([42,42],[0,18],color="black") 
  
    # Création des 6m
    plt.plot([23,37],[10,10],color="black")
    plt.plot([23,23],[0,10],color="black")
    plt.plot([37,37],[0,10],color="black")    
    
    # Création du point de penalty
    PenSpot = plt.Circle((30,14),0.5,color="black")
    ax.add_patch(PenSpot)
    
    # Création des demi-arcs
    BoxArc = Arc((30,18),height=10,width=10,angle=90,theta1=270,theta2=90,color="black")
    CentreArc = Arc((30,90),height=18,width=18,angle=90,theta1=90,theta2=280,color="black")
    ax.add_patch(BoxArc)
    ax.add_patch(CentreArc)
    
    # Suppression des axes
    plt.axis('off')
    
    # Ajout des joueurs
    if tactique == "4-4-2":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(15, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(35, 30, compo["defenseur_central"][1], fontsize = 15)
        # Défenseur lateraux
        plt.text(5, 35, compo["defenseur_lateral"][0], fontsize = 15)
        plt.text(50, 35, compo["defenseur_lateral"][1], fontsize = 15)
        # Milieux défensifs
        plt.text(15, 45, compo["milieu_def"][0], fontsize = 15)
        plt.text(35, 45, compo["milieu_def"][1], fontsize = 15)
        # Milieux offensifs
        plt.text(5, 55, compo["milieu_off"][0], fontsize = 15)
        plt.text(50, 55, compo["milieu_off"][1], fontsize = 15)
        # Attaquants
        plt.text(15, 75, compo["attaque"][0], fontsize = 15)
        plt.text(35, 75, compo["attaque"][1], fontsize = 15)

    elif tactique == "4-3-3":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(15, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(35, 30, compo["defenseur_central"][1], fontsize = 15)
        # Défenseur lateraux
        plt.text(5, 35, compo["defenseur_lateral"][0], fontsize = 15)
        plt.text(50, 35, compo["defenseur_lateral"][1], fontsize = 15)
        # Milieux défensifs
        plt.text(25, 45, compo["milieu_def"][0], fontsize = 15)
        # Milieux offensifs
        plt.text(18, 55, compo["milieu_off"][0], fontsize = 15)
        plt.text(35, 55, compo["milieu_off"][1], fontsize = 15)
        # Attaquants
        plt.text(10, 80, compo["attaque"][0], fontsize = 15)
        plt.text(25, 70, compo["attaque"][1], fontsize = 15)         
        plt.text(40, 80, compo["attaque"][2], fontsize = 15)
        
    elif tactique == "4-1-2-3":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(15, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(35, 30, compo["defenseur_central"][1], fontsize = 15)
        # Défenseur lateraux
        plt.text(5, 35, compo["defenseur_lateral"][0], fontsize = 15)
        plt.text(50, 35, compo["defenseur_lateral"][1], fontsize = 15)
        # Milieux défensifs
        plt.text(25, 45, compo["milieu_def"][0], fontsize = 15)
        # Milieux offensifs
        plt.text(18, 55, compo["milieu_off"][0], fontsize = 15)
        plt.text(35, 55, compo["milieu_off"][1], fontsize = 15)
        # Attaquants
        plt.text(10, 80, compo["attaque"][0], fontsize = 15)
        plt.text(25, 70, compo["attaque"][1], fontsize = 15)         
        plt.text(40, 80, compo["attaque"][2], fontsize = 15)
        
    elif tactique == "4-2-1-3":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(15, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(35, 30, compo["defenseur_central"][1], fontsize = 15)
        # Défenseur lateraux
        plt.text(5, 35, compo["defenseur_lateral"][0], fontsize = 15)
        plt.text(50, 35, compo["defenseur_lateral"][1], fontsize = 15)
        # Milieux défensifs
        plt.text(18, 45, compo["milieu_def"][0], fontsize = 15)
        plt.text(35, 45, compo["milieu_def"][1], fontsize = 15)        
        # Milieux offensifs
        plt.text(28, 55, compo["milieu_off"][0], fontsize = 15)
        # Attaquants
        plt.text(10, 80, compo["attaque"][0], fontsize = 15)
        plt.text(25, 70, compo["attaque"][1], fontsize = 15)         
        plt.text(40, 80, compo["attaque"][2], fontsize = 15)        

    elif tactique == "3-5-2":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(12, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(27, 26, compo["defenseur_central"][1], fontsize = 15)        
        plt.text(42, 30, compo["defenseur_central"][2], fontsize = 15)
        # Milieux défensifs
        plt.text(18, 45, compo["milieu_def"][0], fontsize = 15)
        plt.text(35, 45, compo["milieu_def"][1], fontsize = 15)        
        # Milieux offensifs
        plt.text(10, 60, compo["milieu_off"][0], fontsize = 15)
        plt.text(28, 60, compo["milieu_off"][1], fontsize = 15)
        plt.text(45, 60, compo["milieu_off"][2], fontsize = 15)        
        # Attaquants
        plt.text(15, 75, compo["attaque"][0], fontsize = 15)
        plt.text(35, 75, compo["attaque"][1], fontsize = 15)
         
    elif tactique == "5-3-2":
        # Gardien
        plt.text(26, 3, compo["gardien"][0], fontsize = 15)
        # Défenseur centraux
        plt.text(12, 30, compo["defenseur_central"][0], fontsize = 15)
        plt.text(27, 26, compo["defenseur_central"][1], fontsize = 15)        
        plt.text(42, 30, compo["defenseur_central"][2], fontsize = 15)
        # Défenseur lateraux
        plt.text(5, 35, compo["defenseur_lateral"][0], fontsize = 15)
        plt.text(50, 35, compo["defenseur_lateral"][1], fontsize = 15)
        # Milieux défensifs
        plt.text(18, 45, compo["milieu_def"][0], fontsize = 15)
        plt.text(35, 45, compo["milieu_def"][1], fontsize = 15)        
        # Milieux offensifs
        plt.text(28, 60, compo["milieu_off"][0], fontsize = 15)       
        # Attaquants
        plt.text(15, 75, compo["attaque"][0], fontsize = 15)
        plt.text(35, 75, compo["attaque"][1], fontsize = 15)  
           
    return plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
#%% Mise en page de la page Fonction : Composition d'équipe
if page == pages[8] : 
  st.write("## Fonction : Composition d'équipe")
  st.markdown("<div style='text-align: justify;'>Une fois l'effectif constituée, nous proposons à l'utilisateur une fonction lui permettant de choisir automatiquement ses titulaires pour le prochain match. Elle prend en arguement un effectif et une tactique.</div>", unsafe_allow_html=True)
  st.markdown("*Les effectifs montrés ici sont les résultats directs de notre fonction mercato, indépendamment du budget restant et des suggestions d'achats complémentaires. Des enchères plus basses peuvent être necessaires pour obtenir certaines équipes*.")
  option1 = st.selectbox('Choix du modèle', new_choix)
  col1, col2, col3 = st.columns(3)
  choix1 = ["Toutes", "Ligue 1", "Ligue 2", "Liga", "Premier League", "Serie A"]
  choix2 = ["4-4-2", "4-3-3", "4-1-2-3", "4-2-1-3", "3-5-2", "5-3-2"]
  choix3 = ["4-4-2", "4-3-3", "4-1-2-3", "4-2-1-3", "3-5-2", "5-3-2"]
 
  with col1:
      option2 = st.selectbox('Ligue', choix1)
  with col2:
      option3 = st.selectbox('Tactique mercato', choix2)
  with col3:
      option4 = st.selectbox('Tactique match', choix3)
      
  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(all_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(all_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(all_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(all_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(all_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(all_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(all_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(all_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(all_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(all_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(all_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(all_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(all_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(all_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(all_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(all_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(all_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(all_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(all_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(all_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(all_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(all_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(all_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(all_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(all_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(all_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(all_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(all_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(all_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(all_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(all_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(all_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(all_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(all_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(all_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(all_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(all_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(all_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(all_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(all_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(all_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(all_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(all_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(all_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(all_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(all_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(all_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(all_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(all_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(all_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(all_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Toutes") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(all_532_DTC, "5-3-2")             

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(ligue1_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(ligue1_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(ligue1_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(ligue1_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(ligue1_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(ligue1_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue1_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue1_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(ligue1_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 1") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(ligue1_532_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(ligue2_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(ligue2_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(ligue2_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(ligue2_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(ligue2_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(ligue2_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(ligue2_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(ligue2_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(ligue2_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Ligue 2") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(ligue2_532_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(liga_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(liga_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(liga_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(liga_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(liga_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(liga_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(liga_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(liga_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(liga_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(liga_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(liga_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(liga_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(liga_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(liga_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(liga_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(liga_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(liga_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(liga_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(liga_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(liga_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(liga_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(liga_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(liga_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(liga_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(liga_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(liga_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(liga_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(liga_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(liga_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(liga_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(liga_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(liga_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(liga_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(liga_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(liga_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(liga_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(liga_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(liga_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(liga_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(liga_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(liga_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(liga_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(liga_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(liga_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(liga_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(liga_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(liga_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(liga_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(liga_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(liga_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(liga_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Liga") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(liga_532_DTC, "5-3-2")  
      
  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(premierleague_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(premierleague_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(premierleague_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(premierleague_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(premierleague_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(premierleague_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(premierleague_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(premierleague_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(premierleague_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Premier League") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(premierleague_532_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(serieA_442_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_442_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_442_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_442_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_442_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_442_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-4-2"):
      compo = composition(serieA_442_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_442_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_442_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_442_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_442_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-4-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_442_DTC, "5-3-2")
      
  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(serieA_433_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_433_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_433_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_433_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_433_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_433_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-4-2"):
      compo = composition(serieA_433_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_433_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_433_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_433_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_433_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-3-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_433_DTC, "5-3-2")     
      
  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(serieA_4123_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_4123_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_4123_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_4123_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_4123_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_4123_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-4-2"):
      compo = composition(serieA_4123_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_4123_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_4123_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_4123_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_4123_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-1-2-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_4123_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(serieA_4213_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_4213_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_4213_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_4213_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_4213_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_4213_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-4-2"):
      compo = composition(serieA_4213_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-3-3"): 
      compo = composition(serieA_4213_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_4213_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_4213_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "3-5-2"): 
      compo = composition(serieA_4213_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "4-2-1-3") & (option4 == "5-3-2"): 
      compo = composition(serieA_4213_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(serieA_352_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_352_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_352_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_352_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_352_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_352_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-4-2"):
      compo = composition(serieA_352_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_352_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_352_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_352_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_352_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "3-5-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_352_DTC, "5-3-2")

  if (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(serieA_532_LR, "4-4-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_532_LR, "4-3-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_532_LR, "4-1-2-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_532_LR, "4-2-1-3")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_532_LR, "3-5-2")
  elif (option1 == 'Régression Logistique') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_532_LR, "5-3-2")
      
  if (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-4-2"):
      compo = composition(serieA_532_DTC, "4-4-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-3-3"): 
      compo = composition(serieA_532_DTC, "4-3-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-1-2-3"): 
      compo = composition(serieA_532_DTC, "4-1-2-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "4-2-1-3"): 
      compo = composition(serieA_532_DTC, "4-2-1-3")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "3-5-2"): 
      compo = composition(serieA_532_DTC, "3-5-2")
  elif (option1 == 'Arbre de Décission') & (option2 == "Serie A") & (option3 == "5-3-2") & (option4 == "5-3-2"): 
      compo = composition(serieA_532_DTC, "5-3-2")          
          
  tactique = option4
  
  fig, ax = plt.subplots()    
  fig = compo_viz(compo, tactique)
  st.pyplot(fig)

