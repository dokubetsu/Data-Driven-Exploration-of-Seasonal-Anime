import streamlit as st
import requests
import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import hstack
import time
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Seasonal Anime Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# --- Load Environment Variables ---
load_dotenv()

# --- Database Configuration ---
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '5432')

# Check if credentials are set
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    st.error("Database credentials (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD) not found in environment variables or .env file. Cannot connect to database.")
    st.stop()

# --- Database Connection Function ---
@st.cache_resource
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        st.success("Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Database connection error: {e}")
        return None
    except Exception as e:
         st.error(f"An unexpected error occurred during database connection: {e}")
         return None

# --- Data Fetching Functions ---

@st.cache_data(ttl=300)
def get_latest_scrape_timestamp():
    """Fetches the most recent scrape timestamp from the database."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(scrape_timestamp) FROM anime_seasonal_data_history;")
            latest_timestamp = cur.fetchone()[0]
        return latest_timestamp
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching latest timestamp: {error}")
        return None

@st.cache_data(ttl=3600)
def get_anime_titles(_latest_timestamp):
    """Fetches distinct anime titles for the latest scrape."""
    conn = get_db_connection()
    if conn is None or _latest_timestamp is None: return []
    try:
        query = """
        SELECT DISTINCT title
        FROM anime_seasonal_data_history
        WHERE scrape_timestamp = %s
        ORDER BY title;
        """
        with conn.cursor() as cur:
            cur.execute(query, (_latest_timestamp,))
            titles = [row[0] for row in cur.fetchall()]
        return titles
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching anime titles: {error}")
        return []

@st.cache_data
def load_latest_snapshot(_latest_timestamp):
    """Loads data from the database for the latest scrape timestamp, joining genres/themes."""
    conn = get_db_connection()
    if conn is None or _latest_timestamp is None:
        st.error("Cannot load data: No database connection or timestamp available.")
        return None

    st.write(f"Loading data for timestamp: {_latest_timestamp}...")
    try:
        # Updated query to JOIN tables and aggregate genres/themes
        query = """
        SELECT
            h.scrape_id, -- Include scrape_id for potential future use/debugging
            h.title, h.synopsis, h.studio, h.source, h.demographic,
            h.score, h.members, h.start_date, h.scrape_timestamp,
            h.season, h.year, h.episodes, h.status, h.duration, h.rating,
            h.popularity_rank, h.score_rank,
            -- Aggregate Genres: Group by history record, join genres, combine names
            STRING_AGG(DISTINCT g.name, ', ') AS genres,
            -- Aggregate Themes: Group by history record, join themes, combine names
            STRING_AGG(DISTINCT t.name, ', ') AS themes
        FROM anime_seasonal_data_history h
        -- Left join to include anime even if they have no genres
        LEFT JOIN anime_genres ag ON h.scrape_id = ag.anime_history_scrape_id
        LEFT JOIN genres g ON ag.genre_id = g.genre_id
        -- Left join to include anime even if they have no themes
        LEFT JOIN anime_themes ath ON h.scrape_id = ath.anime_history_scrape_id
        LEFT JOIN themes t ON ath.theme_id = t.theme_id
        WHERE h.scrape_timestamp = %s
        GROUP BY
            -- Group by all columns from the history table to get one row per anime history entry
            h.scrape_id, h.title, h.synopsis, h.studio, h.source, h.demographic,
            h.score, h.members, h.start_date, h.scrape_timestamp,
            h.season, h.year, h.episodes, h.status, h.duration, h.rating,
            h.popularity_rank, h.score_rank
        ORDER BY
            h.title; -- Optional: order results
        """
        df = pd.read_sql_query(query, conn, params=(_latest_timestamp,))

        # --- Data Cleaning/Preparation ---
        # (Most type conversions remain the same)
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['members'] = pd.to_numeric(df['members'], errors='coerce').fillna(0).astype(int)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['scrape_timestamp'] = pd.to_datetime(df['scrape_timestamp'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').astype('Int64')
        df['popularity_rank'] = pd.to_numeric(df['popularity_rank'], errors='coerce').astype('Int64')
        df['score_rank'] = pd.to_numeric(df['score_rank'], errors='coerce').astype('Int64')

        # Fill NA for categorical/string types
        # Genres/Themes might be NULL from STRING_AGG if no matches, so fillna is important
        string_cols = [
            'title', 'synopsis', 'studio', 'source', 'genres', 'themes', # Include genres/themes here
            'demographic', 'season', 'status', 'duration', 'rating'
        ]
        for col in string_cols:
             # Fill actual None/NaN from DB/aggregation with 'N/A'
             df[col] = df[col].fillna('N/A')
             # Convert to string type just in case
             df[col] = df[col].astype(str)
             # Replace any empty strings that might have slipped through
             df[col] = df[col].replace('', 'N/A')

        # --- Derived Columns (remain the same) ---
        df['Broadcast Day'] = df['start_date'].dropna().dt.day_name()
        df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()) if pd.notna(x) and x != 'N/A' else 0)

        # --- Sentiment Analysis (remains the same) ---
        @st.cache_data
        def get_sentiment(text):
            if pd.isna(text) or not text or text == 'N/A': return 0
            try: return TextBlob(str(text)).sentiment.polarity
            except Exception: return 0
        df['Sentiment'] = df['synopsis'].apply(get_sentiment)

        st.success(f"Successfully loaded {len(df)} records from the latest snapshot (with aggregated genres/themes).")
        return df

    except (Exception, psycopg2.Error) as error:
        st.error(f"Error loading latest snapshot data: {error}")
        # Optionally print the query that failed for debugging
        # print("--- Failing Query ---")
        # print(query)
        # print("--- Parameters ---")
        # print((_latest_timestamp,))
        # print("-------------------")
        return None

@st.cache_data
def get_anime_history(title):
    """Fetches score and member history for a specific anime title."""
    conn = get_db_connection()
    if conn is None or not title:
        return pd.DataFrame()

    st.write(f"Fetching history for {title}...")
    try:
        query = """
        SELECT scrape_timestamp, score, members
        FROM anime_seasonal_data_history
        WHERE title = %s AND score IS NOT NULL AND members IS NOT NULL
        ORDER BY scrape_timestamp ASC;
        """
        df_history = pd.read_sql_query(query, conn, params=(title,))

        # Ensure types
        df_history['scrape_timestamp'] = pd.to_datetime(df_history['scrape_timestamp'])
        df_history['score'] = pd.to_numeric(df_history['score'])
        df_history['members'] = pd.to_numeric(df_history['members']).astype(int)

        st.success(f"Loaded {len(df_history)} historical records for {title}.")
        return df_history

    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching history for {title}: {error}")
        return pd.DataFrame()

# --- TF-IDF Calculation (Original - kept for potential other uses or comparison) ---
@st.cache_data
def calculate_tfidf(synopses_series):
    """Calculates TF-IDF matrix for synopses."""
    synopses = synopses_series.fillna('').astype(str)
    if synopses.empty:
        st.warning("Synopsis data is empty. Cannot calculate TF-IDF.")
        return None, None

    st.write("Calculating TF-IDF Matrix...")
    custom_stopwords = list(STOPWORDS) + ['mal', 'source', 'synopsis', 'written', 'anime', 'story', 'life', 'world', 'N/A', 'yet']
    tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=1000, min_df=2)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
        return tfidf_matrix, tfidf_vectorizer
    except ValueError as e:
        if "empty vocabulary" in str(e):
            st.warning("TF-IDF Error: Vocabulary is empty after applying stop words or min_df. Skipping recommendations.")
        else:
            st.warning(f"TF-IDF Error: {e}. Skipping recommendations.")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error during TF-IDF calculation: {e}")
        return None, None

# --- Recommendation Helper Functions (New/Modified) ---

@st.cache_data # Cache the computed similarity matrix
def compute_combined_similarity(df, synopsis_weight=0.7):
    """Computes cosine similarity based on weighted combined TF-IDF of synopsis and genres/themes."""
    if df is None or df.empty:
        return None, None

    st.write(f"Calculating combined similarity matrix (Synopsis Weight: {synopsis_weight*100:.0f}%, Genre/Theme Weight: {(1-synopsis_weight)*100:.0f}%)...")
    genre_theme_weight = 1.0 - synopsis_weight

    # Combine Genres and Themes into a single string for TF-IDF
    df['genre_theme_text'] = df['genres'].fillna('').str.replace(',', ' ') + ' ' + df['themes'].fillna('').str.replace(',', ' ')

    # TF-IDF for Synopsis
    tfidf_synopsis = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix_synopsis = tfidf_synopsis.fit_transform(df['synopsis'].fillna(''))

    # TF-IDF for Genres/Themes
    tfidf_genre_theme = TfidfVectorizer(token_pattern=r'[^, ]+')
    tfidf_matrix_genre_theme = tfidf_genre_theme.fit_transform(df['genre_theme_text'])

    # Combine TF-IDF matrices with weights
    try:
        # Apply weights before stacking
        # Ensure weights sum to 1 (or close enough) for interpretability if needed, though scaling matters more here.
        weighted_synopsis = tfidf_matrix_synopsis * synopsis_weight
        weighted_genre_theme = tfidf_matrix_genre_theme * genre_theme_weight

        combined_features = hstack([weighted_synopsis, weighted_genre_theme]).tocsr()
    except ValueError:
        st.warning("Could not combine features (one matrix might be empty). Falling back to synopsis only.")
        if tfidf_matrix_synopsis.shape[0] > 0:
             combined_features = tfidf_matrix_synopsis # No weight needed if only one
        else:
             st.error("Both synopsis and genre/theme features are empty. Cannot compute similarity.")
             return None, None
    except Exception as e:
        st.error(f"Error combining weighted features: {e}")
        return None, None

    # Compute cosine similarity on the combined matrix
    cosine_sim = linear_kernel(combined_features, combined_features)

    # Create a mapping from anime title to index
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    st.success("Combined similarity matrix calculated.")
    return cosine_sim, indices

def get_recommendations(title, df, cosine_sim_matrix, indices_map):
    """Gets top N recommendations for a given title based on the combined similarity matrix."""
    if df is None or cosine_sim_matrix is None or indices_map is None or title not in indices_map:
        return pd.DataFrame(), [] # Return empty DataFrame and empty list for common tags

    try:
        idx = indices_map[title]
    except KeyError:
        st.error(f"Title '{title}' not found in the index map for recommendations.")
        return pd.DataFrame(), []

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar (excluding itself)
    anime_indices = [i[0] for i in sim_scores]

    # Get details of recommended anime
    recommendations_df = df.loc[anime_indices, ['title', 'score', 'genres', 'themes']].copy()
    recommendations_df['similarity'] = [round(s[1], 3) for s in sim_scores]
    recommendations_df = recommendations_df.sort_values('similarity', ascending=False)

    # --- Explain Recommendations: Find common genres/themes ---
    try:
        source_genres = set(g.strip() for g in df.loc[idx, 'genres'].split(',') if g.strip() and g.strip() != 'N/A')
        source_themes = set(t.strip() for t in df.loc[idx, 'themes'].split(',') if t.strip() and t.strip() != 'N/A')
        source_tags = source_genres.union(source_themes)

        common_tags_list = []
        for rec_idx in anime_indices: # Iterate through recommended anime indices
            rec_title = df.loc[rec_idx, 'title']
            rec_genres = set(g.strip() for g in df.loc[rec_idx, 'genres'].split(',') if g.strip() and g.strip() != 'N/A')
            rec_themes = set(t.strip() for t in df.loc[rec_idx, 'themes'].split(',') if t.strip() and t.strip() != 'N/A')
            rec_tags = rec_genres.union(rec_themes)
            # Find intersection
            common = list(source_tags.intersection(rec_tags))
            common_tags_list.append({"title": rec_title, "common": common})
    except Exception as e:
        st.warning(f"Could not determine common genres/themes for explanation: {e}")
        common_tags_list = [] # Ensure it's an empty list on error

    return recommendations_df, common_tags_list

# --- Plotting Functions (Define them here, outside main) ---
def plot_histogram(series, title, xlabel, ylabel, bins=20):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)

def plot_countplot_y(data, y_col, title, xlabel, ylabel, palette='viridis'):
    fig, ax = plt.subplots(figsize=(12, 8))
    order = data[y_col].value_counts().index
    sns.countplot(data=data, y=y_col, order=order, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)

def plot_barplot_y(index, values, title, xlabel, ylabel, palette='viridis', figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y=index, x=values, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)

def plot_barplot_x(index, values, title, xlabel, ylabel, palette='viridis', color=None, figsize=(10, 6), rotation=45, log_scale=False):
    fig, ax = plt.subplots(figsize=figsize)
    if color:
        sns.barplot(x=index, y=values, color=color, ax=ax)
    else:
        sns.barplot(x=index, y=values, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    if log_scale and (pd.Series(values) > 0).all():
        ax.set_yscale('log')
        ax.set_ylabel(f"{ylabel} (Log Scale)")
    plt.tight_layout()
    st.pyplot(fig)

def plot_regplot(data, x_col, y_col, title, xlabel, ylabel, log_x=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = data[(data[x_col] > 0) & (data[y_col] > 0)]
    if not plot_data.empty:
        sns.regplot(x=x_col, y=y_col, data=plot_data, scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale('log')
            ax.set_xlabel(f"{xlabel} (Log Scale)")
        ax.grid(True, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write(f"Not enough data for {title}.")

def plot_heatmap(data, title, xlabel, ylabel, cmap='viridis', annot=True, fmt=',d', figsize=(12,10)):
     fig, ax = plt.subplots(figsize=figsize)
     sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5, ax=ax)
     ax.set_title(title)
     ax.set_xlabel(xlabel)
     ax.set_ylabel(ylabel)
     plt.tight_layout()
     st.pyplot(fig)

def plot_lineplot(data, x_col, y_col, hue_col, title, xlabel, ylabel, log_y=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, marker='o', palette='Set2', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y and (data[y_col] > 0).all():
        ax.set_yscale('log')
        ax.set_ylabel(f"{ylabel} (Log Scale)")
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

def plot_wordcloud(text, title, stopwords=None, width=1000, height=500, background_color='white', colormap='viridis'):
     if text:
         wordcloud = WordCloud(width=width, height=height, background_color=background_color,
                               stopwords=stopwords, collocations=False, colormap=colormap).generate(text)
         fig, ax = plt.subplots(figsize=(12, 6))
         ax.imshow(wordcloud, interpolation='bilinear')
         ax.axis('off')
         ax.set_title(title)
         st.pyplot(fig)
     else:
         st.write(f"Not enough text for {title} word cloud.")

def plot_countplot_x(data, x_col, title, xlabel, ylabel, palette='viridis', rotation=45, figsize=(10, 6)):
    """Plots a countplot with x-axis categories."""
    fig, ax = plt.subplots(figsize=figsize)
    order = data[x_col].value_counts().index
    sns.countplot(data=data, x=x_col, order=order, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    plt.tight_layout()
    st.pyplot(fig)

# --- Tab Function Definitions ---

# Tab 1: Overview
def tab1_overview(df_snapshot):
    st.header("📊 Basic Info & Distributions (Latest Snapshot)")
    st.metric("Total Anime in Snapshot", df_snapshot.shape[0])

    with st.expander("View Latest Snapshot Data"):
        st.dataframe(df_snapshot)

    col_info, col_nulls = st.columns(2)
    with col_info:
        st.write("Data Types:")
        st.dataframe(df_snapshot.dtypes.astype(str))
    with col_nulls:
        st.write("Null Values:")
        st.dataframe(df_snapshot.isnull().sum())

    st.header("Distributions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Score Distribution")
        plot_histogram(df_snapshot[df_snapshot['score'] > 0]['score'], 'Distribution of Anime Scores (Score > 0)', 'Rating', 'Frequency')

        st.subheader("Popularity Rank Distribution")
        plot_histogram(df_snapshot[df_snapshot['popularity_rank'].notna() & (df_snapshot['popularity_rank'] > 0)]['popularity_rank'], 'Distribution of Popularity Ranks', 'Popularity Rank', 'Frequency', bins=30)

        st.subheader("Episode Count Distribution")
        plot_histogram(df_snapshot[df_snapshot['episodes'].notna() & (df_snapshot['episodes'] > 0)]['episodes'], 'Distribution of Episode Counts (Episodes > 0)', 'Number of Episodes', 'Frequency', bins=30)

    with col2:
        st.subheader("Demographic Distribution")
        plot_countplot_y(df_snapshot[df_snapshot['demographic'] != 'N/A'], 'demographic', 'Number of Anime by Demographic', 'Number of Anime', 'Demographic', palette='pastel')

        st.subheader("Source Distribution")
        plot_countplot_y(df_snapshot[df_snapshot['source'] != 'N/A'], 'source', 'Number of Anime by Source', 'Number of Anime', 'Source', palette='crest')

        st.subheader("Airing Status Distribution")
        plot_countplot_y(df_snapshot[df_snapshot['status'] != 'N/A'], 'status', 'Number of Anime by Airing Status', 'Number of Anime', 'Status', palette='Set2')

    with col3:
        st.subheader("Content Rating Distribution")
        plot_countplot_x(df_snapshot[df_snapshot['rating'] != 'N/A'], 'rating', 'Number of Anime by Content Rating', 'Content Rating', 'Number of Anime', palette='cubehelix', rotation=70, figsize=(10, 7))

        st.subheader("Anime per Broadcast Day")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 'Broadcast Day' in df_snapshot.columns:
            broadcast_day_counts = df_snapshot['Broadcast Day'].dropna().value_counts().reindex(day_order, fill_value=0)
            if not broadcast_day_counts.empty:
                plot_barplot_x(broadcast_day_counts.index, broadcast_day_counts.values,
                                 'No. of Anime Airing Each Day', 'Day Of The Week', 'Number of Anime', palette='crest', figsize=(10, 5))
            else: st.write("No broadcast day data available.")
        else:
            st.write("Broadcast day data not available.")

    st.subheader("Genre Distribution")
    genre_series = df_snapshot['genres'].dropna().str.split(', ', expand=True).stack()
    genre_series = genre_series[genre_series != 'N/A'].str.strip()
    genre_counts = genre_series[genre_series != ''].value_counts()
    if not genre_counts.empty:
        plot_barplot_y(genre_counts.index[:30], genre_counts.values[:30], 'Top 30 Most Common Genres', 'Number of Anime', 'Genre')
    else:
        st.write("No genre data to display.")

    st.subheader("Theme Distribution")
    theme_series = df_snapshot['themes'].dropna().str.split(', ', expand=True).stack()
    theme_series = theme_series[theme_series != 'N/A'].str.strip()
    theme_counts = theme_series[theme_series != ''].value_counts()
    if not theme_counts.empty:
        plot_barplot_y(theme_counts.index[:30], theme_counts.values[:30], 'Top 30 Most Common Themes', 'Number of Anime', 'Theme', figsize=(15,10))
    else:
        st.write("No theme data to display.")

# Tab 2: Time Series
def tab2_time_series(conn):
    st.header("📅 Anime Score & Member History")

    latest_timestamp = get_latest_scrape_timestamp()
    if latest_timestamp is None:
        st.warning("Could not determine the latest data timestamp. Time series analysis unavailable.")
        st.stop()

    available_titles = get_anime_titles(latest_timestamp)
    if not available_titles:
        st.warning("No anime titles found for the latest snapshot.")
        st.stop()

    selected_title = st.selectbox("Select an Anime Title:", available_titles, key="ts_select_main") # Unique key

    if selected_title:
        df_history = get_anime_history(selected_title)

        if not df_history.empty and len(df_history) > 1:
            st.subheader(f"History for: {selected_title}")

            # Calculate Change Metrics
            df_history = df_history.sort_values('scrape_timestamp')
            latest_data = df_history.iloc[-1]
            first_data = df_history.iloc[0]
            previous_data = df_history.iloc[-2]

            score_change_vs_prev = latest_data['score'] - previous_data['score']
            member_change_vs_prev = latest_data['members'] - previous_data['members']
            score_change_vs_first = latest_data['score'] - first_data['score']
            member_change_vs_first = latest_data['members'] - first_data['members']
            time_since_prev = latest_data['scrape_timestamp'] - previous_data['scrape_timestamp']
            time_total = latest_data['scrape_timestamp'] - first_data['scrape_timestamp']

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Score Change (since {previous_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')})",
                          value=f"{latest_data['score']:.2f}", delta=f"{score_change_vs_prev:.2f}")
                st.metric(label=f"Score Change (Total - since {first_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')})",
                          value=f"{latest_data['score']:.2f}", delta=f"{score_change_vs_first:.2f}")
            with col2:
                st.metric(label=f"Members Change (since {previous_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')})",
                          value=f"{latest_data['members']:,}", delta=f"{member_change_vs_prev:,}")
                st.metric(label=f"Members Change (Total - since {first_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')})",
                          value=f"{latest_data['members']:,}", delta=f"{member_change_vs_first:,}")
            st.caption(f"Comparing latest data ({latest_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')}) with previous ({previous_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')}, {time_since_prev}) and first ({first_data['scrape_timestamp'].strftime('%Y-%m-%d %H:%M')}, {time_total} total duration).")
            st.markdown("---")

            # Plotting Score and Members
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:red'
            ax1.set_xlabel('Scrape Timestamp')
            ax1.set_ylabel('Score', color=color)
            ax1.plot(df_history['scrape_timestamp'], df_history['score'], color=color, marker='o', linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Members', color=color)
            ax2.plot(df_history['scrape_timestamp'], df_history['members'], color=color, marker='x', linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()
            plt.title(f'Score and Member Count Over Time for {selected_title}', pad=20)
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif not df_history.empty and len(df_history) <= 1:
             st.warning(f"Only one data point found for {selected_title}. Cannot show trends or changes yet.")
        else:
            st.warning(f"No historical data found for '{selected_title}'.")

# Tab 3: Member Details
def tab3_details(df_snapshot):
    st.header("🎬 Details & Member Relationships (Snapshot)")

    # Filter data for plots
    df_membered = df_snapshot[df_snapshot['members'] > 0]
    df_ranked_mem = df_snapshot[df_snapshot['popularity_rank'].notna() & (df_snapshot['popularity_rank'] > 0) & (df_snapshot['members'] > 0)]

    # Explode Genres/Themes
    df_genres_exploded_members = df_snapshot.dropna(subset=['genres']).assign(Genre=df_snapshot['genres'].str.split(', ')).explode('Genre')
    df_genres_exploded_members = df_genres_exploded_members[(df_genres_exploded_members['Genre'].notna()) & (df_genres_exploded_members['Genre'].str.strip() != '') & (df_genres_exploded_members['Genre'].str.strip() != 'N/A') & (df_genres_exploded_members['members'] > 0)]
    df_themes_exploded_members = df_snapshot.dropna(subset=['themes']).assign(Theme=df_snapshot['themes'].str.split(', ')).explode('Theme')
    df_themes_exploded_members = df_themes_exploded_members[(df_themes_exploded_members['Theme'].notna()) & (df_themes_exploded_members['Theme'].str.strip() != '') & (df_themes_exploded_members['Theme'].str.strip() != 'N/A') & (df_themes_exploded_members['members'] > 0)]

    # Row 1: Members vs Score, Members vs Popularity Rank
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Members vs. Score")
        plot_regplot(df_membered[df_membered['score'] > 0], 'members', 'score', 'Members vs. Score (Log Scale for Members)', 'Number of Members Interested', 'Score', log_x=True)
    with col1b:
        st.subheader("Members vs. Popularity Rank")
        plot_regplot(df_ranked_mem, 'members', 'popularity_rank', 'Members vs. Popularity Rank (Log Scale for Members)', 'Number of Members Interested', 'Popularity Rank', log_x=True)

    # Row 2: Total Members vs Source, Total Members vs Status
    col2a, col2b = st.columns(2)
    with col2a:
        st.subheader("Total Members vs. Source")
        valid_sources_mem = df_membered[(df_membered['source'].notna()) & (df_membered['source'] != 'N/A')]
        if not valid_sources_mem.empty:
            total_members_by_source = valid_sources_mem.groupby('source')['members'].sum().sort_values(ascending=False)
            if not total_members_by_source.empty:
                 plot_barplot_x(total_members_by_source.index, total_members_by_source.values,
                                  'Total Members by Source', 'Source', 'Total Members', palette='plasma', figsize=(10, 5), log_scale=True)
            else: st.write("No source member counts to display.")
        else: st.write("No valid source data with members > 0.")
    with col2b:
        st.subheader("Total Members vs. Airing Status")
        valid_status_mem = df_membered[(df_membered['status'].notna()) & (df_membered['status'] != 'N/A')]
        if not valid_status_mem.empty:
            total_members_by_status = valid_status_mem.groupby('status')['members'].sum().sort_values(ascending=False)
            if not total_members_by_status.empty:
                 plot_barplot_x(total_members_by_status.index, total_members_by_status.values,
                                  'Total Members by Airing Status', 'Status', 'Total Members', palette='Set2', figsize=(10, 6), log_scale=True)
            else: st.write("No status member counts to display.")
        else: st.write("No valid status data with members > 0.")

    # Row 3: Total Members vs Broadcast Day, Total Members vs Demographic
    col3a, col3b = st.columns(2)
    with col3a:
        st.subheader("Total Members vs. Broadcast Day")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 'Broadcast Day' in df_snapshot.columns:
            valid_days_mem = df_membered[df_membered['Broadcast Day'].notna()]
            if not valid_days_mem.empty:
                members_by_day = valid_days_mem.groupby('Broadcast Day')['members'].sum().reindex(day_order, fill_value=0)
                if members_by_day.sum() > 0:
                     plot_barplot_x(members_by_day.index, members_by_day.values,
                                      'Total Members vs Day of Broadcast', 'Day of the Week', 'Number of Members Interested',
                                      palette='flare', figsize=(10, 6), log_scale=True)
                else: st.write("No broadcast day member counts > 0 to display.")
            else: st.write("No valid broadcast day data with members > 0.")
        else: st.write("Broadcast day data not available.")
    with col3b:
        st.subheader("Total Members vs. Demographic")
        valid_demos_mem = df_membered[(df_membered['demographic'].notna()) & (df_membered['demographic'] != 'N/A')]
        if not valid_demos_mem.empty:
            total_members_by_demographic = valid_demos_mem.groupby('demographic')['members'].sum().sort_values(ascending=False)
            if not total_members_by_demographic.empty:
                 plot_barplot_x(total_members_by_demographic.index, total_members_by_demographic.values,
                                  'Total Members by Demographic', 'Demographic', 'Total Members',
                                  palette='gist_earth', figsize=(10, 6), log_scale=True)
            else: st.write("No demographic member counts to display.")
        else: st.write("No valid demographic data with members > 0.")

    st.subheader("Total Members vs. Genre (Top 30)")
    if not df_genres_exploded_members.empty:
        total_members_by_genre = df_genres_exploded_members.groupby('Genre')['members'].sum().sort_values(ascending=False)
        if not total_members_by_genre.empty:
            plot_barplot_y(total_members_by_genre.index[:30], total_members_by_genre.values[:30],
                             'Total Number of Members by Genre (Top 30)', 'Number of Members Interested', 'Genre', palette='magma', figsize=(12, 8))
            st.caption("X-axis (Members) uses a linear scale in the plot above.")
        else: st.write("No genre member counts to display.")
    else: st.write("No valid genre data with members > 0.")

    st.subheader("Total Members vs. Theme (Top 30)")
    if not df_themes_exploded_members.empty:
        total_members_by_theme = df_themes_exploded_members.groupby('Theme')['members'].sum().sort_values(ascending=False)
        if not total_members_by_theme.empty:
            plot_barplot_y(total_members_by_theme.index[:30], total_members_by_theme.values[:30],
                             'Total Number of Members by Theme (Top 30)', 'Number of Members Interested', 'Theme', palette='viridis', figsize=(12, 10))
            st.caption("X-axis (Members) uses a linear scale in the plot above.")
        else: st.write("No theme member counts to display.")
    else: st.write("No valid theme data with members > 0.")

# Tab 4: Score Analysis
def tab4_analysis(df_snapshot):
    st.header("📈 Score Relationships & Analysis (Snapshot)")

    # Explode Genres/Themes
    df_genres_exploded = df_snapshot.dropna(subset=['genres']).assign(Genre=df_snapshot['genres'].str.split(', ')).explode('Genre')
    df_genres_exploded = df_genres_exploded[df_genres_exploded['Genre'].notna() & (df_genres_exploded['Genre'].str.strip() != '') & (df_genres_exploded['Genre'].str.strip() != 'N/A')]
    df_themes_exploded = df_snapshot.dropna(subset=['themes']).assign(Theme=df_snapshot['themes'].str.split(', ')).explode('Theme')
    df_themes_exploded = df_themes_exploded[df_themes_exploded['Theme'].notna() & (df_themes_exploded['Theme'].str.strip() != '') & (df_themes_exploded['Theme'].str.strip() != 'N/A')]

    # Filter data for plots
    df_scored = df_snapshot[df_snapshot['score'] > 0]
    df_ranked = df_snapshot[df_snapshot['popularity_rank'].notna() & (df_snapshot['popularity_rank'] > 0) & (df_snapshot['score'] > 0)]

    # Row 1: Score vs Pop Rank, Score vs Score Rank
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Score vs. Popularity Rank")
        plot_regplot(df_ranked, 'popularity_rank', 'score', 'Score vs. Popularity Rank', 'Popularity Rank', 'Score', log_x=True)
    with col1b:
        st.subheader("Score vs. Score Rank")
        df_score_ranked = df_snapshot[df_snapshot['score_rank'].notna() & (df_snapshot['score_rank'] > 0) & (df_snapshot['score'] > 0)]
        plot_regplot(df_score_ranked, 'score_rank', 'score', 'Score vs. Score Rank', 'Score Rank', 'Score', log_x=True)

    # Row 2: Avg Score vs Studio, Avg Score vs Source
    col2a, col2b = st.columns(2)
    with col2a:
        st.subheader("Avg Score vs. Studio")
        valid_studios = df_scored[(df_scored['studio'].notna()) & (df_scored['studio'] != 'N/A')]
        if not valid_studios.empty:
            average_rating_by_studio = valid_studios.groupby('studio')['score'].mean().sort_values(ascending=False).head(20)
            if not average_rating_by_studio.empty:
                plot_barplot_x(average_rating_by_studio.index, average_rating_by_studio.values,
                                 'Top 20 Studios by Average Score (Score > 0)', 'Studio', 'Average Rating', palette='coolwarm', figsize=(10, 7))
            else: st.write("No average studio scores to display.")
        else: st.write("No valid studio data with scores > 0.")
    with col2b:
        st.subheader("Avg Score vs. Source")
        valid_sources = df_scored[(df_scored['source'].notna()) & (df_scored['source'] != 'N/A')]
        if not valid_sources.empty:
            average_rating_by_source = valid_sources.groupby('source')['score'].mean().sort_values(ascending=False)
            if not average_rating_by_source.empty:
                plot_barplot_x(average_rating_by_source.index, average_rating_by_source.values,
                                 'Average Score by Source (Score > 0)', 'Source', 'Average Rating', palette='viridis', figsize=(10, 5))
            else: st.write("No average source scores to display.")
        else: st.write("No valid source data with scores > 0.")

    # Row 3: Avg Score vs Genre, Avg Score vs Theme
    col3a, col3b = st.columns(2)
    with col3a:
        st.subheader("Avg Score vs. Genre")
        if not df_genres_exploded.empty:
            average_rating_by_genre = df_genres_exploded[df_genres_exploded['score'] > 0].groupby('Genre')['score'].mean().sort_values(ascending=False)
            if not average_rating_by_genre.empty:
                plot_barplot_y(average_rating_by_genre.index[:30], average_rating_by_genre.values[:30],
                                 'Average Score by Genre (Top 30, Score > 0)', 'Average Score', 'Genre', palette='mako')
            else: st.write("No average genre scores to display.")
        else: st.write("No valid genre data to analyze score.")
    with col3b:
        st.subheader("Avg Score vs. Theme")
        if not df_themes_exploded.empty:
            average_rating_by_theme = df_themes_exploded[df_themes_exploded['score'] > 0].groupby('Theme')['score'].mean().sort_values(ascending=False)
            if not average_rating_by_theme.empty:
                plot_barplot_y(average_rating_by_theme.index[:30], average_rating_by_theme.values[:30],
                                 'Average Score by Theme (Top 30, Score > 0)', 'Average Score', 'Theme', palette='rocket')
            else: st.write("No average theme scores to display.")
        else: st.write("No valid theme data to analyze score.")

    # Row 4: Avg Score vs Demographic, Avg Score vs Status
    col4a, col4b = st.columns(2)
    with col4a:
        st.subheader("Avg Score vs. Demographic")
        valid_demos = df_scored[(df_scored['demographic'].notna()) & (df_scored['demographic'] != 'N/A')]
        if not valid_demos.empty:
            average_rating_by_demographic = valid_demos.groupby('demographic')['score'].mean().sort_values(ascending=False)
            if not average_rating_by_demographic.empty:
                plot_barplot_x(average_rating_by_demographic.index, average_rating_by_demographic.values,
                                 'Average Score by Demographic (Score > 0)', 'Demographic', 'Average Score', palette='ocean', figsize=(10, 6))
            else: st.write("No average demographic scores to display.")
        else: st.write("No valid demographic data with scores > 0.")
    with col4b:
        st.subheader("Avg Score vs. Airing Status")
        valid_status = df_scored[(df_scored['status'].notna()) & (df_scored['status'] != 'N/A')]
        if not valid_status.empty:
             average_rating_by_status = valid_status.groupby('status')['score'].mean().sort_values(ascending=False)
             if not average_rating_by_status.empty:
                 plot_barplot_x(average_rating_by_status.index, average_rating_by_status.values,
                                  'Average Score by Airing Status (Score > 0)', 'Status', 'Average Score', palette='Set3', figsize=(10, 6))
             else: st.write("No average status scores to display.")
        else: st.write("No valid status data with scores > 0.")

# Tab 5: Recommendations
def tab5_recommendations(df_snapshot):
    st.header("💡 Find Similar Anime")

    if df_snapshot is None or df_snapshot.empty:
        st.warning("Data not loaded. Cannot generate recommendations.")
        st.stop()

    # Add Weight Slider to Sidebar
    st.sidebar.subheader("Recommendation Weights")
    synopsis_weight = st.sidebar.slider(
        "Weight for Synopsis Similarity", 0.0, 1.0, 0.7, 0.1,
        help="Adjust the importance of synopsis vs. genres/themes. Higher means synopsis matters more."
    )
    st.sidebar.caption(f"Synopsis: {synopsis_weight*100:.0f}%, Genres/Themes: {(1-synopsis_weight)*100:.0f}%")

    # Compute similarity matrix
    cosine_sim, indices = compute_combined_similarity(df_snapshot, synopsis_weight=synopsis_weight)

    if cosine_sim is None or indices is None:
        st.error("Failed to compute similarity matrix. Recommendations unavailable.")
        st.stop()

    valid_titles = indices.index.tolist()
    if not valid_titles:
        st.warning("No valid titles found in the data for recommendations.")
        st.stop()

    selected_anime = st.selectbox("Select an Anime:", valid_titles, key="combo_reco_select_main") # Unique key

    if st.button("Get Recommendations", key="combo_reco_button_main"): # Unique key
        if selected_anime:
            recommendations_df, common_tags_list = get_recommendations(selected_anime, df_snapshot, cosine_sim, indices)

            if not recommendations_df.empty:
                st.subheader(f"Top 10 Recommendations similar to '{selected_anime}':")
                st.dataframe(recommendations_df, use_container_width=True)

                # Display Explanation
                if common_tags_list:
                    st.subheader("Why these recommendations?")
                    with st.expander("Show common Genres/Themes"):
                        try:
                             idx = indices[selected_anime]
                             source_genres = df_snapshot.loc[idx, 'genres']
                             source_themes = df_snapshot.loc[idx, 'themes']
                             st.write(f"**{selected_anime} Genres:** {source_genres}")
                             st.write(f"**{selected_anime} Themes:** {source_themes}")
                             st.markdown("--- ")
                        except Exception:
                             st.write("Could not retrieve source tags.")

                        for item in common_tags_list:
                            rec_title = item['title']
                            common_list = item['common']
                            if common_list:
                                st.write(f"- **{rec_title}:** Common tags: {', '.join(common_list)}")
                            else:
                                st.write(f"- **{rec_title}:** No common genres/themes found (similarity likely based on synopsis).")
            else:
                st.warning(f"Could not find recommendations for '{selected_anime}'.")
        else:
            st.warning("Please select an anime.")

# Tab 6: Word Clouds
def tab6_wordclouds(df_snapshot):
    st.header("🌐 Text Analysis & Word Clouds (Snapshot)")

    # Calculate TF-IDF Matrix
    tfidf_matrix, tfidf_vectorizer = calculate_tfidf(df_snapshot['synopsis'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Synopsis Word Cloud")
        all_synopsis_text = " ".join(df_snapshot['synopsis'].dropna().astype(str)) # Ensure string conversion
        custom_stopwords = list(STOPWORDS) + ['mal', 'source', 'synopsis', 'written', 'anime', 'story', 'life', 'world', 'N/A', 'yet']
        plot_wordcloud(all_synopsis_text, 'Word Cloud of Anime Synopses', stopwords=custom_stopwords, background_color='white')

        st.subheader("Top Keywords (TF-IDF)")
        if tfidf_matrix is not None and tfidf_vectorizer is not None:
             features = tfidf_vectorizer.get_feature_names_out()
             if features.size > 0:
                 tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
                 keyword_freq = tfidf_df.sum().sort_values(ascending=False)
                 st.dataframe(keyword_freq.head(20))

                 st.subheader("Top 30 Keywords Plot")
                 sum_tfidf = keyword_freq.head(30)
                 plot_barplot_x(sum_tfidf.index, sum_tfidf.values, 'Top 30 Keywords by Summed TF-IDF Score',
                                  'Keyword', 'Total TF-IDF Score', color='skyblue', figsize=(12, 6))
             else:
                  st.write("TF-IDF vocabulary is empty, cannot display keywords.")
        else:
            st.write("TF-IDF calculation failed or yielded no results.")

    with col2:
        st.subheader("Title Word Cloud")
        all_titles_text = " ".join(df_snapshot['title'].dropna().astype(str)) # Ensure string conversion
        plot_wordcloud(all_titles_text, 'Word Cloud of Anime Titles', background_color='black', colormap='Pastel1')

        st.subheader("Synopsis Sentiment Distribution")
        plot_histogram(df_snapshot['Sentiment'], 'Distribution of Synopsis Sentiment Scores', 'Sentiment Polarity', 'Frequency')
        st.dataframe(df_snapshot[['title', 'Sentiment']].sort_values(by='Sentiment', ascending=False))


    st.header("Title Length Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Avg Members by Title Word Count")
        word_count_members = df_snapshot[df_snapshot['members'] > 0].groupby('title_word_count')['members'].mean().reset_index()
        if not word_count_members.empty:
            fig_tl_mem, ax_tl_mem = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=word_count_members, x='title_word_count', y='members', marker='o', ax=ax_tl_mem)
            ax_tl_mem.set_title('Average Number of Members by Title Word Count')
            ax_tl_mem.set_xlabel('Number of Words in Title')
            ax_tl_mem.set_ylabel('Average Number of Members')
            ax_tl_mem.grid(True, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_tl_mem)
        else: st.write("Not enough data for title length vs members plot.")

    with col4:
        st.subheader("Avg Title Word Count by Source")
        valid_source_title = df_snapshot[df_snapshot['source'].notna() & (df_snapshot['source'] != 'N/A')]
        if not valid_source_title.empty:
            source_word_count = valid_source_title.groupby('source')['title_word_count'].mean().reset_index().sort_values(by='title_word_count', ascending=False)
            if not source_word_count.empty:
                plot_barplot_x(source_word_count['source'], source_word_count['title_word_count'],
                                 'Average Title Word Count by Source', 'Source', 'Average Title Word Count',
                                 palette='viridis', figsize=(10, 6))
            else: st.write("No source data for title length analysis.")
        else:
            st.write("No valid source data for title length analysis.")

# Tab 7: Genre/Theme Explorer
def tab7_genre_theme_explorer(df_snapshot, conn):
    st.header("🎭 Genre & Theme Explorer")
    st.markdown("Explore anime based on their assigned genres and themes.")

    if df_snapshot is None or df_snapshot.empty:
        st.warning("Snapshot data not loaded. Cannot explore genres/themes.")
        st.stop()
    if conn is None:
        st.warning("Database connection not available. Cannot fetch genre/theme lists.")
        st.stop()

    # Fetch distinct genres and themes from the database for selection
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM genres ORDER BY name;")
            all_genres = ["<Select Genre>"] + [row[0] for row in cur.fetchall()] # Add placeholder
            cur.execute("SELECT name FROM themes ORDER BY name;")
            all_themes = ["<Select Theme>"] + [row[0] for row in cur.fetchall()] # Add placeholder
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching genre/theme list from database: {error}")
        all_genres = ["<Select Genre>"]
        all_themes = ["<Select Theme>"]

    tag_options = ["<Select Tag Type>", "Genre", "Theme"]
    selected_tag_type = st.selectbox("Select Tag Type:", tag_options, key="tag_type_select_main") # Unique key

    selected_tag = None
    if selected_tag_type == "Genre":
        if len(all_genres) <= 1: # Only placeholder exists
            st.warning("No genres found in the database.")
            st.stop()
        selected_tag = st.selectbox("Select Genre:", all_genres, key="genre_select_main") # Unique key
    elif selected_tag_type == "Theme":
        if len(all_themes) <= 1: # Only placeholder exists
            st.warning("No themes found in the database.")
            st.stop()
        selected_tag = st.selectbox("Select Theme:", all_themes, key="theme_select_main") # Unique key

    # Process only if a valid tag type and specific tag are chosen (and not placeholder)
    if selected_tag and selected_tag not in ["<Select Genre>", "<Select Theme>"]:
        st.subheader(f"Exploring {selected_tag_type}: {selected_tag}")

        # Filter the main dataframe
        try:
            # Use word boundaries () to match whole words only
            pattern = f'\b{selected_tag}\b'
            if selected_tag_type == "Genre":
                filtered_df = df_snapshot[df_snapshot['genres'].str.contains(pattern, regex=True, na=False)]
            else: # Theme
                filtered_df = df_snapshot[df_snapshot['themes'].str.contains(pattern, regex=True, na=False)]
        except Exception as e:
             st.error(f"Error filtering DataFrame for tag '{selected_tag}': {e}")
             filtered_df = pd.DataFrame() # Ensure it's an empty DF on error

        if filtered_df.empty:
            st.info(f"No anime found with the {selected_tag_type} '{selected_tag}' in the current snapshot.")
        else:
            st.metric(f"Anime Found with '{selected_tag}'", len(filtered_df))

            col1, col2 = st.columns(2)
            with col1:
                avg_score = filtered_df['score'].mean()
                st.metric(f"Average Score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
            with col2:
                avg_members = filtered_df['members'].mean()
                st.metric(f"Average Members", f"{avg_members:,.0f}" if pd.notna(avg_members) else "N/A")

            st.markdown("--- ")
            st.subheader(f"Anime with {selected_tag_type}: {selected_tag}")
            st.dataframe(filtered_df[['title', 'score', 'members', 'genres', 'themes', 'season', 'year']].sort_values('score', ascending=False), use_container_width=True)

            # Co-occurring Tags Analysis
            st.markdown("--- ")
            st.subheader("Most Common Co-occurring Tags")
            all_tags_list = []
            tag_col_to_exclude = 'genres' if selected_tag_type == "Genre" else 'themes'
            tag_col_to_include = 'themes' if selected_tag_type == "Genre" else 'genres'

            try:
                for idx, row in filtered_df.iterrows():
                    # Get tags from the column *not* being filtered, split and clean
                    included_tags = [t.strip() for t in row[tag_col_to_include].split(',') if t.strip() and t.strip() != 'N/A']
                    all_tags_list.extend(included_tags)
                    # Get tags from the column being filtered, exclude the selected tag, split and clean
                    excluded_tags = [t.strip() for t in row[tag_col_to_exclude].split(',') if t.strip() and t.strip() != selected_tag and t.strip() != 'N/A']
                    all_tags_list.extend(excluded_tags)
            except Exception as e:
                 st.warning(f"Error processing co-occurring tags: {e}")
                 all_tags_list = []

            if all_tags_list:
                tag_counts = pd.Series(all_tags_list).value_counts()
                st.bar_chart(tag_counts.head(15))
                with st.expander("View all co-occurring tag counts"):
                    st.dataframe(tag_counts)
            else:
                st.info("No other tags found co-occurring with the selection.")

            # Score Distribution Plot
            st.markdown("--- ")
            st.subheader("Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(filtered_df['score'].dropna(), kde=True, bins=15, ax=ax)
            ax.set_title(f'Score Distribution for Anime with {selected_tag_type}: {selected_tag}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

# --- Main Execution Logic ---
def main():
    st.title("📈 Seasonal Anime Analysis Dashboard")

    # --- Sidebar Setup ---
    st.sidebar.title("Anime Analysis Options")
    st.sidebar.markdown("Analysis based on the latest data snapshot.")

    conn = get_db_connection()
    if conn is None:
        st.error("Halting execution due to database connection failure.")
        st.stop()

    latest_timestamp = get_latest_scrape_timestamp()
    if latest_timestamp:
        # Use strftime for better formatting if it's a datetime object
        ts_str = latest_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(latest_timestamp, datetime.datetime) else str(latest_timestamp)
        st.sidebar.success(f"Latest data from: {ts_str}")
    else:
        st.sidebar.warning("Could not retrieve latest data timestamp.")
        st.stop()

    df_snapshot = load_latest_snapshot(latest_timestamp)

    if df_snapshot is None or df_snapshot.empty:
        st.error("Failed to load the latest data snapshot. Cannot proceed.")
        st.stop()
    else:
         st.sidebar.info(f"Loaded {len(df_snapshot)} anime records.")

    # --- Tab Creation ---
    tab_titles = [
        "📊 Overview",             # Calls tab1_overview
        "📈 Score Analysis",        # Calls tab4_analysis
        "🎬 Member Details",       # Calls tab3_details
        "🌐 Text & Word Clouds",   # Calls tab6_wordclouds
        "💡 Recommendations",      # Calls tab5_recommendations
        "📅 Time Series",          # Calls tab2_time_series
        "🎭 Genre/Theme Explorer"  # Calls tab7_genre_theme_explorer
    ]
    tabs = st.tabs(tab_titles)

    # Assign functions to tabs
    with tabs[0]: tab1_overview(df_snapshot)
    with tabs[1]: tab4_analysis(df_snapshot)       # Score analysis
    with tabs[2]: tab3_details(df_snapshot)        # Member details
    with tabs[3]: tab6_wordclouds(df_snapshot)     # Text analysis
    with tabs[4]: tab5_recommendations(df_snapshot)
    with tabs[5]: tab2_time_series(conn)           # Time series requires conn
    with tabs[6]: tab7_genre_theme_explorer(df_snapshot, conn) # Explorer requires conn

    # --- Footer/Sidebar Info ---
    st.sidebar.markdown("----")
    st.sidebar.info("Dashboard using enriched data from PostgreSQL.")
    # Optional: Close connection (Streamlit usually handles this)
    # conn.close()

# --- Script Entry Point ---
if __name__ == "__main__":
    main() 