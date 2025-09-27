import pandas as pd
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import json
import os
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Sample COLUMN_DEFINITIONS (replace with your actual definitions)
COLUMN_DEFINITIONS = {
    'rainfall_mm_total': {
        'meaning': 'Total annual rainfall in millimeters',
        'categorization': 'Arid: <500 mm, Semi-arid: 500-1000 mm, Sub-humid: 1000-1500 mm, Humid: >1500 mm',
        'classification': 'Influences recharge potential; higher rainfall increases recharge',
        'related_columns': ['ground_water_recharge_ham_total', 'total_ground_water_availability_in_the_area_ham']
    },
    'ground_water_recharge_ham_total': {
        'meaning': 'Total annual groundwater recharge in hectare-meters',
        'categorization': 'Low: <2000 ham, Moderate: 2000-5000 ham, High: >5000 ham',
        'classification': 'Indicates replenishment potential; recharge = sum of canal, irrigation, rainfall contributions',
        'related_columns': ['rainfall_mm_total', 'total_ground_water_availability_in_the_area_ham', 'stage_of_ground_water_extraction_total']
    },
    'ground_water_extraction_for_all_uses_ha_m_total': {
        'meaning': 'Total annual groundwater extraction for all uses in hectare-meters',
        'categorization': 'Low: <1000 ham, Moderate: 1000-4000 ham, High: >4000 ham',
        'classification': 'Indicates usage intensity; high extraction may lead to over-exploitation',
        'related_columns': ['ground_water_recharge_ham_total', 'stage_of_ground_water_extraction_total']
    },
    'total_ground_water_availability_in_the_area_ham': {
        'meaning': 'Total annual net groundwater availability in hectare-meters',
        'categorization': 'Low: <1000 ham, Moderate: 1000-3000 ham, High: >3000 ham',
        'classification': 'Indicates sustainability; availability = recharge - extraction',
        'related_columns': ['ground_water_recharge_ham_total', 'ground_water_extraction_for_all_uses_ha_m_total']
    },
    'stage_of_ground_water_extraction_total': {
        'meaning': 'Percentage of extraction to recharge',
        'categorization': 'Safe: <70%, Semi-critical: 70-90%, Critical: 90-100%, Over-exploited: >100%',
        'classification': 'Indicates sustainability status; stage = (extraction / recharge) * 100',
        'related_columns': ['ground_water_recharge_ham_total', 'ground_water_extraction_for_all_uses_ha_m_total']
    },
    'state': {
        'meaning': 'State name of the area',
        'categorization': 'Categorical (list of states)',
        'classification': 'Provides location context',
        'related_columns': ['district']
    },
    'district': {
        'meaning': 'District name of the area',
        'categorization': 'Categorical (list of districts)',
        'classification': 'Provides location context',
        'related_columns': ['state']
    }
}

# Load data from SQLite database
def load_data():
    db_path = r'C:\Users\dr-prashant\Downloads\sih\INGRES_db.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM my_table"  # Updated table name
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Columns in loaded DataFrame:", df.columns.tolist())
    if df.empty:
        raise ValueError("DataFrame is empty. Check the database or query.")
    return df

# Initialize embedding model
try:
    embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# NLP for column selection
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_relevant_columns(query):
    """
    Identifies direct and immediate related columns only, avoiding recursive related columns.
    Returns a list of relevant column names.
    """
    tokens = word_tokenize(query.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalnum()]
    query_text = ' '.join(lemmatized_tokens)
    
    # Direct matches (keyword or semantic)
    column_texts = [COLUMN_DEFINITIONS[col]['meaning'] for col in COLUMN_DEFINITIONS]
    column_names = list(COLUMN_DEFINITIONS.keys())
    query_embedding = embed_model.encode([query_text], show_progress_bar=False)
    column_embeddings = embed_model.encode(column_texts, show_progress_bar=False)
    
    similarities = cosine_similarity(query_embedding, column_embeddings)[0]
    threshold = 0.3
    direct_columns = [column_names[i] for i, sim in enumerate(similarities) if sim > threshold]
    
    for col in COLUMN_DEFINITIONS:
        col_tokens = [lemmatizer.lemmatize(word) for word in re.split(r'[_-]', col.lower()) if word not in stop_words]
        if any(token in lemmatized_tokens for token in col_tokens):
            if col not in direct_columns:
                direct_columns.append(col)
    
    # Add immediate related columns only
    final_columns = set(direct_columns)
    for col in direct_columns:
        related = COLUMN_DEFINITIONS.get(col, {}).get('related_columns', [])
        final_columns.update([r for r in related if r in COLUMN_DEFINITIONS])
    
    # Add contextual columns
    base_columns = ['state', 'district'] if 'state' in df.columns and 'district' in df.columns else []
    final_columns.update(base_columns)
    
    return list(final_columns) or list(COLUMN_DEFINITIONS.keys())[:5]

# Create text chunks with only relevant columns
def create_text_chunk(row, relevant_columns):
    chunks = []
    for col in relevant_columns:
        if col in row and col in COLUMN_DEFINITIONS:
            value = row[col] if pd.notna(row[col]) else 'N/A'
            defn = COLUMN_DEFINITIONS.get(col, {})
            meaning = defn.get('meaning', 'No meaning available')
            categorization = defn.get('categorization', 'No categorization available')
            classification = defn.get('classification', 'No classification available')
            chunks.append(f"{col.replace('_', ' ').title()}: {value} (Meaning: {meaning}; Categorization: {categorization}; Classification: {classification})")
    return ' | '.join(chunks) if chunks else "No relevant data available"

# Main batch creation function
def create_batches():
    # Load data
    try:
        global df
        df = load_data()
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return
    
    # Check for required columns
    required_columns = list(COLUMN_DEFINITIONS.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in DataFrame: {missing_columns}")
        for missing_col in missing_columns:
            similar_cols = [col for col in df.columns if missing_col.lower() in col.lower()]
            if similar_cols:
                print(f"Suggestions for '{missing_col}': {similar_cols}")
    
    # Clean None values in key columns
    df['state'] = df.get('state', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
    df['district'] = df.get('district', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
    
    # Sample query to determine relevant columns (can be customized)
    sample_query = "how much affect will occur on groundwater if rainfall is 30% more next year?"
    relevant_columns = extract_relevant_columns(sample_query)
    
    # Create text chunks and embeddings
    try:
        df['text'] = df.apply(lambda row: create_text_chunk(row, relevant_columns), axis=1)
        if not df['text'].str.strip().any():
            print("Error: No valid text chunks created. Check DataFrame content.")
            return
        texts = df['text'].tolist()
        embeddings = embed_model.encode(texts, show_progress_bar=True)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return
    
    # Prepare batch data
    batch_data = []
    for i, row in df.iterrows():
        batch_data.append({
            'id': str(i),
            'text': row['text'],
            'metadata': {'state': str(row.get('state', 'Unknown')), 'district': str(row.get('district', 'Unknown'))},
            'embedding': embeddings[i].tolist()
        })
    
    # Save batches to JSON
    output_path = 'gec_batches.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        print(f"Batches saved to {output_path}")
    except Exception as e:
        print(f"Error saving batches: {e}")

if __name__ == "__main__":
    create_batches()