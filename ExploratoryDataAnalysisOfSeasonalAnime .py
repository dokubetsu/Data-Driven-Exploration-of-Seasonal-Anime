#!/usr/bin/env python
# coding: utf-8

# # Analysis Of Seasonal Anime Summer 2024

# ### Imports and Configuration

# In[1]:


import requests
import warnings
import os
import pandas as pd
import numpy as np # Import numpy for NaN handling
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# --- Configuration ---
FORCE_SCRAPE = False # Set to True to force re-scraping even if CSV exists
ANIME_DATA_CSV = 'anime_seasonal_data.csv'
# -------------------


# ### Data Acquisition (Scraping)

# In[2]:


def scrape_seasonal_anime(output_csv_file):
    """Scrapes the MyAnimeList seasonal anime page and saves data to a CSV file."""
    print(f"Scraping MyAnimeList seasonal anime page...")
    url = "https://myanimelist.net/anime/season"
    
    try:
        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return False # Indicate failure

    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    anime_containers = soup.select('.seasonal-anime')

    all_anime_data = []

    for anime in anime_containers:
        title_element = anime.select_one('.h2_anime_title a')
        title = title_element.get_text(strip=True) if title_element else 'N/A'

        synopsis_element = anime.select_one('.synopsis .preline')
        synopsis = synopsis_element.get_text(strip=True) if synopsis_element else 'N/A'

        studio_element = anime.select_one('.property:contains("Studio") .item a')
        studio = studio_element.get_text(strip=True) if studio_element else 'N/A'
        
        source_element = anime.select_one('.property:contains("Source") .item')
        source = source_element.get_text(strip=True) if source_element else 'N/A'

        genres_section = anime.select_one('.genres')
        genres = [genre.get_text(strip=True) for genre in genres_section.select('.genre a')] if genres_section else []

        themes_section = anime.find('span', class_='caption', text='Themes')
        themes = []
        if themes_section:
            themes_parent = themes_section.find_parent('div')
            if themes_parent:
                themes = [theme.get_text(strip=True) for theme in themes_parent.find_all('a')]

        demographic_element = anime.select_one('.property:contains("Demographic") .item a')
        demographic = demographic_element.get_text(strip=True) if demographic_element else 'N/A'

        score_element = anime.select_one('.score-label')
        score = score_element.get_text(strip=True) if score_element else 'N/A'
        
        members_element = anime.select_one('.member')
        members = members_element.get_text(strip=True) if members_element else 'N/A'

        start_date_element = anime.select_one('.js-start_date')
        start_date = start_date_element.get_text(strip=True) if start_date_element else 'N/A'

        anime_data = {
            'Title': title,
            'Synopsis': synopsis,
            'Studio': studio,
            'Source': source,
            'Genres': ", ".join(genres),
            'Themes': ", ".join(themes),
            'Demographic': demographic,
            'Score': score,
            'Members': members,
            'Start Date': start_date,
        }
        all_anime_data.append(anime_data)

    if not all_anime_data:
        print("No anime data scraped.")
        return False # Indicate failure

    # Writing the collected data into a CSV file
    try:
        with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=all_anime_data[0].keys())
            writer.writeheader()
            writer.writerows(all_anime_data)
        print(f"Data successfully scraped and saved to {output_csv_file}")
        return True # Indicate success
    except IOError as e:
        print(f"Error writing to CSV file {output_csv_file}: {e}")
        return False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        return False # Indicate failure


# ### Data Loading and Cleaning

# In[3]:


def convert_to_int(x):
    """Converts member strings (e.g., '1.2K', '1M') to integers."""
    if isinstance(x, (int, float)): # Already numeric
        return int(x)
    x = str(x).strip()
    if x == 'N/A' or not x:
        return 0 # Or np.nan, depending on desired handling
    
    x = x.replace(',', '') 
    if 'K' in x:
        return int(float(x.replace('K', '')) * 1000)
    elif 'M' in x:
        return int(float(x.replace('M', '')) * 1000000)
    else:
        try:
            return int(x)
        except ValueError:
            return 0 # Or np.nan

def load_and_clean_data(filepath):
    """Loads data from CSV and performs initial cleaning."""
    print(f"Loading and cleaning data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return None

    # Convert Start Date - handle potential errors
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce') # Coerce errors to NaT

    # Convert Score - handle 'N/A' and potential errors
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce') # Coerce errors (like 'N/A') to NaN
    df['Score'] = df['Score'].fillna(0) # Fill NaN scores with 0 (or np.nan or mean/median if preferred)

    # Convert Members - use the helper function
    df['Members'] = df['Members'].apply(convert_to_int)
    
    # Handle potential N/A or empty strings in categorical columns if necessary
    categorical_cols = ['Studio', 'Source', 'Demographic']
    for col in categorical_cols:
        df[col] = df[col].fillna('N/A') # Fill NaN with 'N/A' string
        df[col] = df[col].replace('', 'N/A') # Replace empty strings

    # Ensure Genres and Themes are strings, handle NaN before splitting
    df['Genres'] = df['Genres'].fillna('').astype(str) 
    df['Themes'] = df['Themes'].fillna('').astype(str)

    print("Data loaded and cleaned.")
    df.info() # Display info after cleaning
    return df


# ### Main Execution: Get Data

# In[4]:


# Decide whether to scrape or load
if FORCE_SCRAPE or not os.path.exists(ANIME_DATA_CSV):
    success = scrape_seasonal_anime(ANIME_DATA_CSV)
    if not success:
        # Handle scraping failure - maybe exit or try loading anyway?
        print("Scraping failed. Attempting to load data if available...")
        # Decide if you want to exit here:
        # import sys
        # sys.exit("Exiting due to scraping failure.")
        df = load_and_clean_data(ANIME_DATA_CSV) # Try loading anyway
    else:
        df = load_and_clean_data(ANIME_DATA_CSV) # Load the newly scraped data
else:
    print(f"Loading existing data from {ANIME_DATA_CSV} (set FORCE_SCRAPE=True to re-scrape).")
    df = load_and_clean_data(ANIME_DATA_CSV)

# Check if DataFrame loaded successfully before proceeding
if df is None:
     import sys
     sys.exit("Could not load data. Exiting.") # Exit if df failed to load


# ### Exploratory Data Analysis (EDA)

# In[5]: # Keep original In numbers for reference, but they are now just comments


# Basic Info
print("\n--- Basic DataFrame Info ---")
print(f"Shape: {df.shape}")
print("\nNull values check:")
print(df.isnull().sum())


# In[598]:


# Score Distribution
print("\n--- Score Analysis ---")
print(df['Score'].describe())

plt.figure(figsize=(10, 6))
# Filter out scores of 0 if they represent 'N/A' or missing data for plotting distribution
sns.histplot(df[df['Score'] > 0]['Score'], kde=True) 
plt.title('Distribution of Anime Scores (Score > 0)')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[600]: # Genre Analysis


# Genre Analysis (Handle empty strings resulting from fillna)
print("\n--- Genre Analysis ---")
# Split only non-empty strings, filter out empty results
genre_series = df['Genres'].str.split(', ', expand=True).stack()
genre_series = genre_series[genre_series != ''] # Filter out empty strings after split
genre_counts = genre_series.str.strip().value_counts()

print("Unique Genres Found:", genre_counts.index.tolist())
print("\nGenre Counts:")
print(genre_counts)


# In[603]:


plt.figure(figsize=(12, 8)) # Adjusted size
sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='viridis')
plt.title('Most Common Genres')
plt.xlabel('Number of Anime')
plt.ylabel('Genre')
plt.tight_layout() # Add tight layout
plt.show()


# In[604]: # Theme Analysis


# Theme Analysis (Handle empty strings)
print("\n--- Theme Analysis ---")
# Split only non-empty strings, filter out empty results
theme_series = df['Themes'].str.split(', ', expand=True).stack()
theme_series = theme_series[theme_series != ''] # Filter out empty strings after split
theme_counts = theme_series.str.strip().value_counts()

print("Unique Themes Found:", theme_counts.index.tolist())
print("\nTheme Counts:")
print(theme_counts)


# In[606]:


plt.figure(figsize=(15, 10))
sns.barplot(y=theme_counts.index, x=theme_counts.values, palette='viridis')
plt.title('Most Common Themes') # Corrected title
plt.xlabel('Number of Anime')
plt.ylabel('Theme') # Corrected label
plt.tight_layout() # Add tight layout
plt.show()


# In[607]: # Source Analysis


print("\n--- Source Analysis ---")
print(df['Source'].value_counts())


# In[608]: # Studio Analysis


print("\n--- Studio Analysis ---")
print("Top 10 Studios by Anime Count:")
print(df['Studio'].value_counts().head(10))


# In[610]: # Top Anime


print("\n--- Top Anime Analysis ---")
# Filter out N/A or 0 scores before sorting
top_anime = df[df['Score'] > 0].sort_values(by='Score', ascending=False).head(10)

print("\nTop 10 Anime Of The Season (Score > 0):")
if not top_anime.empty:
    for index, row in top_anime.iterrows():
        print(f"Title: {row['Title']}")
        # Print other details as before...
        print(f"Genres: {row['Genres']}") # Genres is now a string again
        print(f"Score: {row['Score']}")
        print(f"Members: {row['Members']}")
        print(f"Demographic: {row['Demographic']}")
        # print(f"Synopis: {row['Synopsis']}") # Optionally uncomment
        print('-' * 40)
else:
    print("No anime found with score > 0.")


# In[611]: # Studio vs Score


print("\n--- Studio vs. Score ---")
# Filter out N/A or 0 scores before grouping
average_rating_by_studio = df[df['Score'] > 0].groupby('Studio')['Score'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=average_rating_by_studio.index, y=average_rating_by_studio.values, palette='coolwarm')
plt.title('Top 10 Studios by Average Score (Score > 0)')
plt.xlabel('Studio')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right') # Improve label readability
plt.tight_layout()
plt.show()


# In[612]: # Demographic Distribution


print("\n--- Demographic Analysis ---")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Demographic', order=df['Demographic'].value_counts().index, palette='pastel')
plt.title('Number of Anime by Demographic')
plt.xlabel('Number of Anime')
plt.ylabel('Demographic')
plt.tight_layout()
plt.show()


# In[613]: # Genre vs Score


print("\n--- Genre vs. Score ---")
# Explode genres again, ensuring Score > 0
df_genres_exploded = df.assign(Genre=df['Genres'].str.split(', ')).explode('Genre')
df_genres_exploded = df_genres_exploded[df_genres_exploded['Genre'] != ''] # Filter empty genres
average_rating_by_genre = df_genres_exploded[df_genres_exploded['Score'] > 0].groupby('Genre')['Score'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(y=average_rating_by_genre.index, x=average_rating_by_genre.values, palette='mako')
plt.title('Average Score by Genre (Score > 0)')
plt.xlabel('Average Score')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[614]: # Theme vs Score


print("\n--- Theme vs. Score ---")
# Explode themes again, ensuring Score > 0
df_themes_exploded = df.assign(Theme=df['Themes'].str.split(', ')).explode('Theme')
df_themes_exploded = df_themes_exploded[df_themes_exploded['Theme'] != ''] # Filter empty themes
average_rating_by_theme = df_themes_exploded[df_themes_exploded['Score'] > 0].groupby('Theme')['Score'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(y=average_rating_by_theme.index, x=average_rating_by_theme.values, palette='rocket') # Changed palette
plt.title('Average Score by Theme (Score > 0)')
plt.xlabel('Average Score')
plt.ylabel('Theme') # Corrected Label
plt.tight_layout()
plt.show()


# In[615]: # Broadcast Day Analysis


print("\n--- Broadcast Day Analysis ---")
# Ensure 'Start Date' is not NaT before extracting day name
df['Broadcast Day'] = df['Start Date'].dropna().dt.day_name() 
# Define order for plotting
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Handle cases where some days might be missing
broadcast_day_counts = df['Broadcast Day'].value_counts().reindex(day_order, fill_value=0)

print("Anime Count per Broadcast Day:")
print(broadcast_day_counts)


# In[617]:


plt.figure(figsize=(10, 5)) # Adjusted size
sns.barplot(x=broadcast_day_counts.index, y=broadcast_day_counts.values, palette='crest') # Use reindexed data
plt.title('No. of Anime Airing Each Day of the Week')
plt.xlabel('Day Of The Week')
plt.ylabel('Number of Anime')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[619]: # Members by Broadcast Day


print("\n--- Members by Broadcast Day ---")
# Group by the ordered day index
members_by_day = df.groupby('Broadcast Day')['Members'].sum().reindex(day_order, fill_value=0)
print("Total Members Interested by Broadcast Day:")
print(members_by_day)


# In[620]:


plt.figure(figsize=(10, 6)) # Adjusted size
sns.barplot(x=members_by_day.index, y=members_by_day.values, palette='flare') # Use reindexed data
plt.title('Total Members vs Day of Broadcast')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Members Interested')
# Consider log scale if differences are huge, but check for zero values first
if (members_by_day > 0).all():
     plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--') # Added grid style
plt.tight_layout()
plt.show()


# In[621]: # Genre vs Members


print("\n--- Genre vs. Members ---")
# Use the previously exploded genre dataframe
total_members_by_genre = df_genres_exploded.groupby('Genre')['Members'].sum().sort_values(ascending=False)

print("Total Number of Members by Genre:")
print(total_members_by_genre)

plt.figure(figsize=(12, 8)) # Adjusted size
sns.barplot(x=total_members_by_genre.values, y=total_members_by_genre.index, palette='magma') # Changed palette
plt.title('Total Number of Members by Genre')
plt.xlabel('Number of Members Interested (Log Scale)') # Label reflects scale
plt.ylabel('Genre')
# Check for non-positive values before applying log scale
if (total_members_by_genre > 0).all():
     plt.xscale('log')
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


# In[622]: # Members vs Score Scatter Plot


print("\n--- Members vs. Score ---")
plt.figure(figsize=(10, 6))
# Filter data for plotting where both members and score > 0
plot_data = df[(df['Members'] > 0) & (df['Score'] > 0)]
if not plot_data.empty:
    sns.regplot(x='Members', y='Score', data=plot_data, scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'red'}) # Adjusted scatter points
    plt.title('Members vs. Score (Log Scale for Members)')
    plt.xlabel('Number of Members Interested (Log Scale)')
    plt.ylabel('Score')
    plt.xscale('log') # Apply log scale to x-axis
    # plt.ylim(top=10) # Optional: Set y-limit if needed
    plt.grid(True, linestyle='--') # Added grid
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data points with Members > 0 and Score > 0 to plot.")


# In[623]: # Source vs Score


print("\n--- Source vs. Score ---")
# Ensure Score > 0
average_rating_by_source = df[df['Score'] > 0].groupby('Source')['Score'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=average_rating_by_source.index, y=average_rating_by_source.values, palette='viridis') # Changed palette
plt.title('Average Score by Source (Score > 0)')
plt.xlabel('Source')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[625]: # Source vs Members


print("\n--- Source vs. Members ---")
total_members_by_source = df.groupby('Source')['Members'].sum().sort_values(ascending=False)
print('Total No. of Members by Source:')
print(total_members_by_source)

plt.figure(figsize=(10, 5))
sns.barplot(x=total_members_by_source.index, y=total_members_by_source.values, palette='plasma') # Changed palette
plt.title('Total Members by Source (Log Scale)')
plt.xlabel('Source')
plt.ylabel('Total Members (Log Scale)')
# Check before applying log scale
if (total_members_by_source > 0).all():
    plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


# In[626]: # Demographic vs Score


print("\n--- Demographic vs. Score ---")
# Ensure Score > 0
average_rating_by_demographic = df[df['Score'] > 0].groupby('Demographic')['Score'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6)) # Adjusted size
sns.barplot(x=average_rating_by_demographic.index, y=average_rating_by_demographic.values, palette='ocean') # Changed palette and orientation
plt.title('Average Score by Demographic (Score > 0)')
plt.xlabel('Demographic') # Corrected label
plt.ylabel('Average Score') # Corrected label
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[627]: # Genre vs Demographic Heatmap


print("\n--- Genre vs. Demographic ---")
# Use previously exploded genre dataframe
genre_demographic_count = df_genres_exploded.groupby(['Genre', 'Demographic']).size().unstack(fill_value=0)

print("Genre vs Demographic Count:")
# print(genre_demographic_count) # Can be large, print optionally

plt.figure(figsize=(12, 10)) # Adjusted size
sns.heatmap(genre_demographic_count, annot=True, fmt='d', cmap='Reds', linewidths=0.5)
plt.title('Heatmap of Anime Count by Genre and Demographic')
plt.xlabel('Demographic')
plt.ylabel('Genre')
# plt.xticks(rotation=45) # Rotation might not be needed if labels fit
plt.tight_layout()
plt.show()


# In[628]: # Theme vs Demographic Heatmap


print("\n--- Theme vs. Demographic ---")
# Use previously exploded theme dataframe
theme_demographic_count = df_themes_exploded.groupby(['Theme', 'Demographic']).size().unstack(fill_value=0)

print("Theme vs Demographic Count:")
# print(theme_demographic_count) # Can be large, print optionally

plt.figure(figsize=(12, 10)) # Adjusted size
sns.heatmap(theme_demographic_count, annot=True, fmt='d', cmap='Blues', linewidths=0.5) # Changed cmap
plt.title('Heatmap of Anime Count by Theme and Demographic')
plt.xlabel('Demographic')
plt.ylabel('Theme')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[629]: # Demographic vs Members


print("\n--- Demographic vs. Members ---")
total_members_by_demographic = df.groupby('Demographic')['Members'].sum().sort_values(ascending=False)
print('Total members for each Demographic:')
print(total_members_by_demographic)

plt.figure(figsize=(10, 6)) # Adjusted size
sns.barplot(x=total_members_by_demographic.index, y=total_members_by_demographic.values, palette='gist_earth') # Changed palette
plt.title('Total Members by Demographic (Log Scale)')
plt.xlabel('Demographic')
plt.ylabel('Total Members (Log Scale)')
# Check before log scale
if (total_members_by_demographic > 0).all():
    plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


# In[630]: # Genre/Demographic vs Members Heatmap


print("\n--- Members by Genre and Demographic ---")
# Use previously exploded genre dataframe
genre_demographic_members = df_genres_exploded.groupby(['Genre', 'Demographic'])['Members'].sum().unstack(fill_value=0)

plt.figure(figsize=(12, 10)) # Adjusted size
# Use annot=True only if heatmap is not too dense, otherwise consider removing it
annot_flag = genre_demographic_members.shape[0] < 20 # Example condition
sns.heatmap(genre_demographic_members, annot=annot_flag, fmt=',d', cmap='YlGnBu', linewidths=0.5) # Added comma formatting
plt.title('Heatmap of Total Members by Genre and Demographic')
plt.xlabel('Demographic')
plt.ylabel('Genre')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[631]: # Theme/Demographic vs Members Heatmap


print("\n--- Members by Theme and Demographic ---")
# Use previously exploded theme dataframe
theme_demographic_members = df_themes_exploded.groupby(['Theme', 'Demographic'])['Members'].sum().unstack(fill_value=0)

plt.figure(figsize=(12, 10)) # Adjusted size
annot_flag = theme_demographic_members.shape[0] < 20 # Example condition
sns.heatmap(theme_demographic_members, annot=annot_flag, fmt=',d', cmap='PuBuGn', linewidths=0.5) # Changed cmap, added formatting
plt.title('Heatmap of Total Members by Theme and Demographic')
plt.xlabel('Demographic')
plt.ylabel('Theme')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[632]: # Demographic/Day vs Members Heatmap


print("\n--- Members by Demographic and Broadcast Day ---")
# Use the ordered day_order for columns if possible
demographic_day_members = df.groupby(['Demographic', 'Broadcast Day'])['Members'].sum().unstack(fill_value=0).reindex(columns=day_order, fill_value=0)

plt.figure(figsize=(10, 8)) # Adjusted size
sns.heatmap(demographic_day_members, annot=True, fmt=',d', cmap='coolwarm', linewidths=0.5) # Changed cmap, added formatting
plt.title('Heatmap of Total Members by Demographic and Broadcast Day')
plt.xlabel('Broadcast Day')
plt.ylabel('Demographic')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[633]: # Line Plot: Members Trend by Demographic


print("\n--- Trend of Members by Demographic Across Broadcast Days ---")
# Use the same grouped data as above
plt.figure(figsize=(12, 7)) # Adjusted size
if not df_grouped.empty:
    sns.lineplot(data=df_grouped, x='Broadcast Day', y='Members', hue='Demographic', marker='o', palette='Set2')
    plt.title('Trend of Members by Demographic Across Broadcast Days')
    plt.xlabel('Broadcast Day')
    plt.ylabel('Total Members')
    # Check for zeros before applying log scale
    if (df_grouped['Members'] > 0).all():
         plt.yscale('log')
         plt.ylabel('Total Members (Log Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Demographic', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to plot member trends.")


# ### Text Analysis (Synopsis)

# In[636]: # TF-IDF Vectorization


print("\n--- Text Analysis: TF-IDF ---")
# Ensure Synopsis is string and handle potential NaN values
df['Synopsis'] = df['Synopsis'].fillna('').astype(str)

# Define custom stop words more robustly
custom_stopwords = list(STOPWORDS) + ['mal', 'source', 'synopsis', 'written', 'anime', 'story', 'life', 'world', 'N/A'] # Added common words

tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=1000, min_df=2) # Adjusted parameters
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Synopsis'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Calculate and display top keywords
    keyword_freq = tfidf_df.sum().sort_values(ascending=False)
    print("Top 10 Keywords from Synopses (TF-IDF):")
    print(keyword_freq.head(10))

    # Plot top keywords
    sum_tfidf = keyword_freq.head(30)
    plt.figure(figsize=(12, 6)) # Adjusted size
    sum_tfidf.plot(kind='bar', color='skyblue')
    plt.title('Top 30 Keywords by Summed TF-IDF Score')
    plt.xlabel('Keyword')
    plt.ylabel('Total TF-IDF Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"TF-IDF Error: {e}. Skipping TF-IDF analysis.")
    tfidf_matrix = None # Ensure tfidf_matrix is defined even on error


# In[640]: # Sentiment Analysis


print("\n--- Text Analysis: Sentiment ---")
def get_sentiment(text):
    if pd.isna(text) or not text:
        return 0 # Return neutral for empty/NaN synopsis
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception as e:
        print(f"Error processing text for sentiment: {text[:50]}... Error: {e}")
        return 0 # Return neutral on error

df['Sentiment'] = df['Synopsis'].apply(get_sentiment)
print("Sentiment Analysis Results (Sample):")
print(df[['Title', 'Sentiment']].head())

# Plot Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment'], bins=20, kde=True)
plt.title('Distribution of Synopsis Sentiment Polarity')
plt.xlabel('Sentiment Polarity (-1: Negative, 1: Positive)')
plt.ylabel('Frequency')
plt.show()


# In[641]: # Word Cloud (Synopsis)


print("\n--- Text Analysis: Word Cloud (Synopsis) ---")
all_synopsis_text = " ".join(df['Synopsis'].dropna()) # Join only non-null synopses
if all_synopsis_text:
    # Use the same extended stopwords as TF-IDF
    wordcloud = WordCloud(width=1000, height=500, background_color='white', stopwords=custom_stopwords, collocations=False).generate(all_synopsis_text) # Added collocations=False
    plt.figure(figsize=(12, 6)) # Adjusted size
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Anime Synopses')
    plt.show()
else:
    print("Not enough text in synopses to generate a word cloud.")


# In[642]: # Word Cloud (Titles)


print("\n--- Text Analysis: Word Cloud (Titles) ---")
all_titles_text = " ".join(df['Title'].dropna())
if all_titles_text:
    # Use default stopwords for titles unless specific ones are needed
    wordcloud_titles = WordCloud(width=1000, height=500, background_color='black', colormap='Pastel1', collocations=False).generate(all_titles_text) # Changed appearance
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_titles, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Anime Titles') # Corrected Title
    plt.show()
else:
    print("Not enough text in titles to generate a word cloud.")


# ### Title Length Analysis

# In[643]:


print("\n--- Title Length Analysis ---")
df['title_word_count'] = df['Title'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
word_count_members = df[df['Members'] > 0].groupby('title_word_count')['Members'].mean().reset_index()

print("Average Members by Title Word Count:")
print(word_count_members)


# In[645]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=word_count_members, x='title_word_count', y='Members', marker='o')
plt.title('Average Number of Members by Title Word Count')
plt.xlabel('Number of Words in Title')
plt.ylabel('Average Number of Members')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


# In[646]:


source_word_count = df.groupby('Source')['title_word_count'].mean().reset_index().sort_values(by='title_word_count', ascending=False) # Sort values

print("\nAverage Title Word Count by Source:")
print(source_word_count)

plt.figure(figsize=(10, 6))
sns.barplot(data=source_word_count, x='Source', y='title_word_count', palette='viridis')
plt.title('Average Title Word Count by Source')
plt.xlabel('Source')
plt.ylabel('Average Title Word Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### Simple Recommendation System (Based on Synopsis)

# In[657]:


print("\n--- Recommendation System ---")
# Ensure tfidf_matrix was created successfully
if tfidf_matrix is not None and tfidf_matrix.shape[0] > 0:
    # --- User Input ---
    # Choose an anime title that exists in the dataset
    available_titles = df['Title'].tolist()
    # Example: Choose the first title if available, otherwise handle error
    if available_titles:
        input_title = available_titles[0] 
        # Or use a specific known title: input_title = 'Shikanoko Nokonoko Koshitantan' 
        print(f"Finding recommendations similar to: '{input_title}'")
    else:
        input_title = None
        print("No titles available in the dataset for recommendations.")
    # ------------------

    if input_title and input_title in available_titles:
        try:
            input_index = df[df['Title'] == input_title].index[0]
            
            # Calculate cosine similarities
            cosine_similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix).flatten()
            
            # Get top N similar indices (excluding the input itself)
            num_recommendations = 5
            # Add 1 because argsort includes the item itself
            similar_indices = cosine_similarities.argsort()[-(num_recommendations + 1):][::-1] 
            # Remove the input index if present
            similar_indices = [idx for idx in similar_indices if idx != input_index][:num_recommendations] 

            if similar_indices:
                recommended_anime = df.iloc[similar_indices]
                print(f"\nTop {num_recommendations} recommendations based on synopsis:")
                for i, row in recommended_anime.iterrows():
                    similarity_score = cosine_similarities[i]
                    print(f"- {row['Title']} (Similarity: {similarity_score:.2f})")
            else:
                print("Could not find sufficiently similar anime.")

        except IndexError:
            print(f"Error: Input title '{input_title}' not found in the DataFrame index.")
        except Exception as e:
            print(f"An error occurred during recommendation generation: {e}")
    elif input_title:
        print(f"Input title '{input_title}' not found in the dataset.")

else:
    print("Skipping recommendation system as TF-IDF matrix is not available.")


# In[ ]: # End of script marker


print("\n--- Analysis Complete ---")


