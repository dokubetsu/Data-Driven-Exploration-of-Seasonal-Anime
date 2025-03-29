
import requests
import warnings
import os
import csv  
import psycopg2
import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv 
import time
import pandas as pd 
import argparse 
import re 
from urllib.parse import urljoin

warnings.filterwarnings('ignore')

load_dotenv()

# --- Database Configuration ---
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '5432') # Default PostgreSQL port


if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    print("Error: Database credentials (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD) not found in environment variables.")
    exit()

def convert_score_to_numeric(score_str):
    """Converts score string to numeric, handling 'N/A'."""
    if score_str == 'N/A' or not score_str:
        return None
    try:
        return float(score_str)
    except ValueError:
        return None

def convert_members_to_int(members_str):
    """Converts member string (e.g., '1.2K', '1M') to integer, handling 'N/A'."""
    if isinstance(members_str, (int, float)): return int(members_str)
    x = str(members_str).strip()
    if x == 'N/A' or not x: return None
    x = x.replace(',', '')
    if 'K' in x:
        try: return int(float(x.replace('K', '')) * 1000)
        except ValueError: return None
    if 'M' in x:
        try: return int(float(x.replace('M', '')) * 1000000)
        except ValueError: return None
    try: return int(x)
    except ValueError: return None

def convert_start_date(date_str):
    """Converts various date string formats to YYYY-MM-DD or None."""
    if not date_str or date_str == 'N/A':
        return None
    try:
        return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce').strftime('%Y-%m-%d')
        except ValueError:
            print(f"Warning: Could not parse date: {date_str}")
            return None

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during DB connection: {e}")
    return conn

def parse_rank(rank_str):
    """Extracts integer rank from strings like '#123', returning None if invalid."""
    if not rank_str or rank_str == 'N/A':
        return None
    match = re.search(r'#(\d+)', rank_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def parse_int_field(value_str):
    """Safely converts a string to an integer, returning None on failure."""
    if not value_str or value_str.lower() in ['n/a', 'unknown', '']:
        return None
    try:
        return int(value_str.strip())
    except (ValueError, TypeError):
        return None

# --- Individual Page Scraping Function ---
def scrape_individual_anime_page(anime_url):
    """Scrapes additional details from an individual MAL anime page."""
    print(f"  Scraping details from: {anime_url} ...")
    details = {
        'episodes': None, 'status': None, 'duration': None,
        'rating': None, 'popularity_rank': None, 'score_rank': None
    }
    try:
        response = requests.get(anime_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        page_soup = BeautifulSoup(response.content, 'html.parser')

        # Find the left sidebar info column
        left_col = page_soup.find('div', class_='leftside')
        if not left_col:
            print(f"  Warning: Could not find left sidebar info for {anime_url}")
            return details

        # Extract details using helper function
        def get_info(label):
            element = left_col.find('span', class_='dark_text', string=lambda t: t and label in t)
            if element and element.parent:
                 value = element.next_sibling
                 if value and isinstance(value, str):
                     cleaned_value = value.strip()
                     if cleaned_value:
                        return cleaned_value
                 if element.find_next_sibling('a'):
                     return element.find_next_sibling('a').get_text(strip=True)
            return 'N/A' 

        details['episodes'] = parse_int_field(get_info('Episodes:'))
        details['status'] = get_info('Status:')
        details['duration'] = get_info('Duration:')
        details['rating'] = get_info('Rating:')
        details['popularity_rank'] = parse_rank(get_info('Popularity:'))
        details['score_rank'] = parse_rank(get_info('Ranked:'))

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching individual anime page {anime_url}: {e}")
    except Exception as e:
        print(f"  Error parsing individual anime page {anime_url}: {e}")

    return details

# --- Main Seasonal Scraping Function ---
def scrape_seasonal_anime(target_year=None, target_season=None):
    """
    Scrapes MAL seasonal anime page for a specific year/season (or current),
    then scrapes individual pages for more details.
    Saves combined data to the PostgreSQL database.
    """
    base_url = "https://myanimelist.net"
    seasonal_path = "/anime/season"
    if target_year and target_season:
        url = f"{base_url}{seasonal_path}/{target_year}/{target_season.lower()}"
        print(f"Scraping specific season: {target_season.capitalize()} {target_year} from {url}")
    else:
        url = f"{base_url}{seasonal_path}"
        print(f"Scraping current season from {url}")

    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching seasonal URL {url}: {e}")
        return False

    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    current_season = target_season
    current_year = target_year
    if not current_year or not current_season:
        page_title = soup.title.string if soup.title else ''
        h1_text = soup.find('h1').get_text(strip=True) if soup.find('h1') else ''
        title_match = re.search(r"(Winter|Spring|Summer|Fall)\s+(\d{4})", page_title, re.IGNORECASE)
        h1_match = re.search(r"(Winter|Spring|Summer|Fall)\s+(\d{4})", h1_text, re.IGNORECASE)

        if title_match:
            current_season = title_match.group(1).lower()
            current_year = int(title_match.group(2))
            print(f"Determined current season/year from title: {current_season.capitalize()} {current_year}")
        elif h1_match:
             current_season = h1_match.group(1).lower()
             current_year = int(h1_match.group(2))
             print(f"Determined current season/year from H1: {current_season.capitalize()} {current_year}")
        else:
             print("Warning: Could not automatically determine current season/year from page. Season/Year columns might be NULL.")

    anime_containers = soup.select('.seasonal-anime')
    if not anime_containers:
        print(f"Could not find anime containers on the page: {url}. Scraping failed for this season/year.")
        return False

    all_anime_data = []
    scrape_timestamp = datetime.datetime.now(datetime.timezone.utc)

    total_anime = len(anime_containers)
    print(f"Found {total_anime} anime entries on seasonal page. Processing...")
    start_time = time.time()

    for i, anime in enumerate(anime_containers):
        print(f"Processing anime {i+1}/{total_anime}...")
        # --- Extract data from seasonal page ---
        title_element = anime.select_one('.h2_anime_title a')
        title = title_element.get_text(strip=True) if title_element else 'N/A'

        anime_page_url = None
        if title_element:
            anime_page_url = urljoin(base_url, title_element.get('href'))

        synopsis_element = anime.select_one('.synopsis .preline')
        synopsis = synopsis_element.get_text(strip=True) if synopsis_element else 'N/A'

        studio_element = anime.select_one('.property:contains("Studio") .item a')
        studio = studio_element.get_text(strip=True) if studio_element else 'N/A'

        source_element = anime.select_one('.property:contains("Source") .item')
        source = source_element.get_text(strip=True) if source_element else 'N/A'

        # --- Get Genre and Theme names as lists --- 
        genres_section = anime.select_one('.genres')
        genre_names = []
        if genres_section:
            genre_names = [g.get_text(strip=True) for g in genres_section.select('.genre a[href*="/genres/"]') if g.get_text(strip=True)]

        themes_section = anime.select_one('.properties')
        theme_names = []
        if themes_section:
            theme_names = [th.get_text(strip=True) for th in themes_section.select('.property .item a[href*="/themes/"]') if th.get_text(strip=True)]

        demographic_element = anime.select_one('.property .item a[href*="/demographic/"]')
        demographic = demographic_element.get_text(strip=True) if demographic_element else 'N/A'

        score_element = anime.select_one('.score-label')
        score_str = score_element.get_text(strip=True) if score_element else 'N/A'
        score = float(score_str) if score_str not in ['N/A', None, ''] else None

        members_element = anime.select_one('.member')
        members_str = members_element.get_text(strip=True) if members_element else 'N/A'
        members = convert_members_to_int(members_str)

        start_date_element = anime.select_one('.js-start_date')
        start_date_str = start_date_element.get_text(strip=True) if start_date_element else 'N/A'
        start_date = convert_start_date(start_date_str)

        # --- Scrape additional details from individual page ---
        individual_details = {
            'episodes': None, 'status': None, 'duration': None,
            'rating': None, 'popularity_rank': None, 'score_rank': None
        }
        if anime_page_url:
            individual_details = scrape_individual_anime_page(anime_page_url)
            print(f"  Pausing for 1 second...")
            time.sleep(1)
        else:
            print(f"  Warning: Could not find individual page URL for {title}")

        # --- Store combined data (excluding raw genre/theme strings) ---
        anime_record = {
            'title': title,
            'synopsis': synopsis,
            'studio': studio,
            'source': source,
            'demographic': demographic,
            'score': score,
            'members': members,
            'start_date': start_date,
            'scrape_timestamp': scrape_timestamp,
            'season': current_season,
            'year': current_year,
            **individual_details
        }
        all_anime_data.append({
            'record': anime_record,
            'genre_names': genre_names,
            'theme_names': theme_names
        })

    if not all_anime_data:
        print("No anime data extracted.")
        return False

    # --- Database Insertion ---
    conn = get_db_connection()
    if not conn:
        print("Database connection failed. Cannot insert data.")
        return False

    inserted_count = 0
    skipped_count = 0
    start_db_time = time.time()

    try:
        with conn.cursor() as cur:
            # SQL queries
            insert_history_sql = """
            INSERT INTO anime_seasonal_data_history (
                title, synopsis, studio, source, demographic, score, members,
                start_date, scrape_timestamp, season, year, episodes,
                status, duration, rating, popularity_rank, score_rank
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (title, scrape_timestamp) DO NOTHING
            RETURNING scrape_id;
            """
            insert_genre_sql = "INSERT INTO genres (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING genre_id;"
            select_genre_sql = "SELECT genre_id FROM genres WHERE name = %s;"
            link_genre_sql = "INSERT INTO anime_genres (anime_history_scrape_id, genre_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;"

            insert_theme_sql = "INSERT INTO themes (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING theme_id;"
            select_theme_sql = "SELECT theme_id FROM themes WHERE name = %s;"
            link_theme_sql = "INSERT INTO anime_themes (anime_history_scrape_id, theme_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;"

            # Process each anime record
            for item in all_anime_data:
                record = item['record']
                genre_names = item['genre_names']
                theme_names = item['theme_names']

                try:
                    # Insert the main history record
                    cur.execute(insert_history_sql, (
                        record['title'], record['synopsis'], record['studio'], record['source'],
                        record['demographic'], record['score'], record['members'],
                        record['start_date'], record['scrape_timestamp'], record['season'], record['year'],
                        record['episodes'], record['status'], record['duration'], record['rating'],
                        record['popularity_rank'], record['score_rank']
                    ))

                    # Get the scrape_id of the inserted row
                    result = cur.fetchone()
                    if result:
                        history_scrape_id = result[0]
                        inserted_count += 1

                        # Process Genres
                        for genre_name in genre_names:
                            cur.execute(insert_genre_sql, (genre_name,))
                            genre_result = cur.fetchone()
                            if genre_result:
                                genre_id = genre_result[0]
                            else:
                                cur.execute(select_genre_sql, (genre_name,))
                                genre_id = cur.fetchone()[0]
                            cur.execute(link_genre_sql, (history_scrape_id, genre_id))

                        # Process Themes
                        for theme_name in theme_names:
                            cur.execute(insert_theme_sql, (theme_name,))
                            theme_result = cur.fetchone()
                            if theme_result:
                                theme_id = theme_result[0]
                            else:
                                cur.execute(select_theme_sql, (theme_name,))
                                theme_id = cur.fetchone()[0]
                            cur.execute(link_theme_sql, (history_scrape_id, theme_id))

                    else:
                        skipped_count += 1
                        print(f"  Skipped duplicate entry for {record['title']} at {record['scrape_timestamp']}")

                except psycopg2.Error as db_err:
                    print(f"Database transaction error for '{record.get('title', 'N/A')}': {db_err}")
                    conn.rollback() # Rollback the transaction for this specific anime
                except Exception as e:
                     print(f"Unexpected error during DB processing for '{record.get('title', 'N/A')}': {e}")
                     conn.rollback()
            
            # Commit the entire batch transaction if no fatal errors occurred
            conn.commit()
            print(f"Database operations completed in {time.time() - start_db_time:.2f} seconds.")

    except (Exception, psycopg2.Error) as error:
        print(f"General Error during database operation: {error}")
        conn.rollback() # Rollback any pending changes
        return False
    finally:
        if conn:
            conn.close()

    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Scraping and processing finished in {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes).")
    print(f"Successfully processed {inserted_count} new anime history records.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} duplicate history entries (title/timestamp). Links were not added for these.")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape MyAnimeList seasonal anime data, including details from individual pages.")
    parser.add_argument("--year", type=int, help="The specific year to scrape (e.g., 2024).")
    parser.add_argument("--season", type=str, choices=['winter', 'spring', 'summer', 'fall'],
                        help="The specific season to scrape (winter, spring, summer, fall).")

    args = parser.parse_args()

    if (args.year and not args.season) or (not args.year and args.season):
        parser.error("--year and --season must be used together.")
        exit(1)

    print("Starting scraper (fetching individual page details will take longer)...")
    success = scrape_seasonal_anime(target_year=args.year, target_season=args.season)

    if success:
        print("Scraper finished successfully.")
    else:
        print("Scraper encountered errors.")
        exit(1)