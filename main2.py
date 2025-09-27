
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Added missing import
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
import warnings
import re
import ollama
import os
import psutil
import traceback
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
    's_no': {
        'meaning': 'Serial number of the record.',
        'categorization': 'No categorization (serial number).',
        'classification': 'No categorization (serial number, not used for classification).',
        'related_columns': []
    },
    'state': {
        'meaning': 'The state where the district is located.',
        'categorization': 'No categorization (categorical: name of Indian state or union territory).',
        'classification': 'No categorization (categorical: name of state, used for context but not direct classification).',
        'related_columns': ['district', 'assessment_unit']
    },
    'district': {
        'meaning': 'The district name.',
        'categorization': 'No categorization (categorical: name of district).',
        'classification': 'No categorization (categorical: name of district, used for context but not direct classification).',
        'related_columns': ['state', 'assessment_unit']
    },
    'assessment_unit': {
        'meaning': 'Assessment unit used for groundwater calculation.',
        'categorization': 'No categorization (categorical: name of block/taluk or assessment unit).',
        'classification': 'No categorization (categorical: name of block/taluk, used to identify the city/assessment unit).',
        'related_columns': ['state', 'district']
    },
    'rainfall_mm_c': {
        'meaning': 'Rainfall in command area in millimeters.',
        'categorization': 'Very Low: <200 mm, Low: 200-500 mm, Moderate: 500-1000 mm, High: 1000-2000 mm, Very High: 2000-3000 mm, Extremely High: >3000 mm.',
        'classification': 'Not a primary classifier. Very Low (<200 mm) may support No Data or Hilly Area if recharge is minimal (<500 ham); High (>2000 mm) supports Safe if extraction is low (stage_of_ground_water_extraction_percent_c ≤70%). If missing, supports No Data.',
        'related_columns': ['rainfall_mm_nc', 'rainfall_mm_pq', 'rainfall_mm_total', 'ground_water_recharge_ham_rainfall_recharge_c']
    },
    'rainfall_mm_nc': {
        'meaning': 'Rainfall in non-command area in millimeters.',
        'categorization': 'Very Low: <200 mm, Low: 200-500 mm, Moderate: 500-1000 mm, High: 1000-2000 mm, Very High: 2000-3000 mm, Extremely High: >3000 mm.',
        'classification': 'Not a primary classifier. Very Low (<200 mm) may support No Data or Hilly Area if recharge is minimal (<500 ham); High (>2000 mm) supports Safe if extraction is low (stage_of_ground_water_extraction_percent_nc ≤70%). If missing, supports No Data.',
        'related_columns': ['rainfall_mm_c', 'rainfall_mm_pq', 'rainfall_mm_total', 'ground_water_recharge_ham_rainfall_recharge_nc']
    },
    'rainfall_mm_pq': {
        'meaning': 'Rainfall in poor quality area in millimeters.',
        'categorization': 'Very Low: <200 mm, Low: 200-500 mm, Moderate: 500-1000 mm, High: 1000-2000 mm, Very High: 2000-3000 mm, Extremely High: >3000 mm.',
        'classification': 'Not a primary classifier. Very Low (<200 mm) may support No Data or Hilly Area; High (>2000 mm) supports Saline if quality_tagging_major_parameter_present_pq indicates “Salinity”. If missing, supports No Data.',
        'related_columns': ['rainfall_mm_c', 'rainfall_mm_nc', 'rainfall_mm_total', 'ground_water_recharge_ham_rainfall_recharge_pq', 'quality_tagging_major_parameter_present_pq']
    },
    'rainfall_mm_total': {
        'meaning': 'Total rainfall across all areas in millimeters.',
        'categorization': 'Very Low: <200 mm, Low: 200-500 mm, Moderate: 500-1000 mm, High: 1000-2000 mm, Very High: 2000-3000 mm, Extremely High: >3000 mm.',
        'classification': 'Not a primary classifier. Very Low (<200 mm) may support No Data or Hilly Area if recharge is minimal (<1000 ham); High (>2000 mm) supports Safe if extraction is low (stage_of_ground_water_extraction_percent_total ≤70%). If missing, supports No Data.',
        'related_columns': ['rainfall_mm_c', 'rainfall_mm_nc', 'rainfall_mm_pq', 'ground_water_recharge_ham_rainfall_recharge_total']
    },
    'total_geographical_area_ha_recharge_worthy_area_ha_c': {
        'meaning': 'Recharge-worthy area in command area (hectares).',
        'categorization': 'Very Low: <5000 ha, Low: 5000-15000 ha, Moderate: 15000-30000 ha, High: 30000-50000 ha, Very High: 50000-100000 ha, Extremely High: >100000 ha.',
        'classification': 'Not a primary classifier. Low (<5000 ha) may support Hilly Area if total_geographical_area_ha_hilly_area >30% of total_geographical_area_ha_total; High (>50000 ha) supports Safe or Semi Critical if recharge is adequate (>10000 ham) and extraction is low (≤90%). If missing or zero, supports No Data.',
        'related_columns': ['total_geographical_area_ha_recharge_worthy_area_ha_nc', 'total_geographical_area_ha_recharge_worthy_area_ha_pq', 'total_geographical_area_ha_recharge_worthy_area_ha_total', 'total_geographical_area_ha_hilly_area']
    },
    'total_geographical_area_ha_recharge_worthy_area_ha_nc': {
        'meaning': 'Recharge-worthy area in non-command area (hectares).',
        'categorization': 'Very Low: <5000 ha, Low: 5000-15000 ha, Moderate: 15000-30000 ha, High: 30000-50000 ha, Very High: 50000-100000 ha, Extremely High: >100000 ha.',
        'classification': 'Not a primary classifier. Low (<5000 ha) may support Hilly Area if total_geographical_area_ha_hilly_area >30% of total_geographical_area_ha_total; High (>50000 ha) supports Safe or Semi Critical if recharge is adequate (>10000 ham) and extraction is low (≤90%). If missing or zero, supports No Data.',
        'related_columns': ['total_geographical_area_ha_recharge_worthy_area_ha_c', 'total_geographical_area_ha_recharge_worthy_area_ha_pq', 'total_geographical_area_ha_recharge_worthy_area_ha_total', 'total_geographical_area_ha_hilly_area']
    },
    'total_geographical_area_ha_recharge_worthy_area_ha_pq': {
        'meaning': 'Recharge-worthy area in poor quality area (hectares).',
        'categorization': 'Very Low: <5000 ha, Low: 5000-15000 ha, Moderate: 15000-30000 ha, High: 30000-50000 ha, Very High: 50000-100000 ha, Extremely High: >100000 ha.',
        'classification': 'Saline: If >50000 ha and quality_tagging_major_parameter_present_pq indicates “Salinity”. Low (<5000 ha) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['total_geographical_area_ha_recharge_worthy_area_ha_c', 'total_geographical_area_ha_recharge_worthy_area_ha_nc', 'total_geographical_area_ha_recharge_worthy_area_ha_total', 'quality_tagging_major_parameter_present_pq']
    },
    'total_geographical_area_ha_recharge_worthy_area_ha_total': {
        'meaning': 'Total recharge-worthy area across all zones (hectares).',
        'categorization': 'Very Low: <10000 ha, Low: 10000-30000 ha, Moderate: 30000-60000 ha, High: 60000-100000 ha, Very High: 100000-150000 ha, Extremely High: >150000 ha.',
        'classification': 'Not a primary classifier. Used as a reference for proportional calculations (e.g., hilly_area ratio). If zero or missing, supports No Data. High (>100000 ha) may support Safe or Semi Critical if recharge is high (>20000 ham).',
        'related_columns': ['total_geographical_area_ha_recharge_worthy_area_ha_c', 'total_geographical_area_ha_recharge_worthy_area_ha_nc', 'total_geographical_area_ha_recharge_worthy_area_ha_pq', 'total_geographical_area_ha_hilly_area']
    },
    'total_geographical_area_ha_hilly_area': {
        'meaning': 'Total hilly area in hectares.',
        'categorization': 'Very Low: <2000 ha, Low: 2000-5000 ha, Moderate: 5000-10000 ha, High: 10000-20000 ha, Very High: 20000-50000 ha, Extremely High: >50000 ha.',
        'classification': 'Hilly Area: If >30% of total_geographical_area_ha_total (e.g., >3000 ha for a 10000 ha total area) and recharge/extraction data is low (<1000 ham) or missing. Otherwise, secondary to stage_of_ground_water_extraction_percent_total. If missing or zero, supports No Data.',
        'related_columns': ['total_geographical_area_ha_total', 'total_geographical_area_ha_recharge_worthy_area_ha_total']
    },
    'total_geographical_area_ha_total': {
        'meaning': 'Total geographical area in hectares.',
        'categorization': 'Very Low: <10000 ha, Low: 10000-30000 ha, Moderate: 30000-60000 ha, High: 60000-100000 ha, Very High: 100000-150000 ha, Extremely High: >150000 ha.',
        'classification': 'Not a primary classifier. Used as a reference for proportional calculations (e.g., hilly_area ratio). If zero or missing, supports No Data. High (>100000 ha) may support Safe or Semi Critical if recharge is high (>20000 ham).',
        'related_columns': ['total_geographical_area_ha_hilly_area', 'total_geographical_area_ha_recharge_worthy_area_ha_total']
    },
    'ground_water_recharge_ham_rainfall_recharge_c': {
        'meaning': 'Groundwater recharge from rainfall in command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. Very Low (<500 ham) may support No Data or Hilly Area; High (>6000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%. If missing or zero, supports No Data.',
        'related_columns': ['rainfall_mm_c', 'ground_water_recharge_ham_rainfall_recharge_nc', 'ground_water_recharge_ham_rainfall_recharge_pq', 'ground_water_recharge_ham_rainfall_recharge_total']
    },
    'ground_water_recharge_ham_rainfall_recharge_nc': {
        'meaning': 'Groundwater recharge from rainfall in non-command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. Very Low (<500 ham) may support No Data or Hilly Area; High (>6000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%. If missing or zero, supports No Data.',
        'related_columns': ['rainfall_mm_nc', 'ground_water_recharge_ham_rainfall_recharge_c', 'ground_water_recharge_ham_rainfall_recharge_pq', 'ground_water_recharge_ham_rainfall_recharge_total']
    },
    'ground_water_recharge_ham_rainfall_recharge_pq': {
        'meaning': 'Groundwater recharge from rainfall in poor quality area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Saline: If >6000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<500 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['rainfall_mm_pq', 'ground_water_recharge_ham_rainfall_recharge_c', 'ground_water_recharge_ham_rainfall_recharge_nc', 'ground_water_recharge_ham_rainfall_recharge_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_rainfall_recharge_total': {
        'meaning': 'Total groundwater recharge from rainfall (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-12000 ham, Very High: 12000-20000 ham, Extremely High: >20000 ham.',
        'classification': 'Not a primary classifier. Very Low (<1000 ham) may support No Data or Hilly Area; High (>12000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%. If missing or zero, supports No Data.',
        'related_columns': ['rainfall_mm_total', 'ground_water_recharge_ham_rainfall_recharge_c', 'ground_water_recharge_ham_rainfall_recharge_nc', 'ground_water_recharge_ham_rainfall_recharge_pq']
    },
    'ground_water_recharge_ham_canals_c': {
        'meaning': 'Recharge from canals in command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_canals_nc', 'ground_water_recharge_ham_canals_pq', 'ground_water_recharge_ham_canals_total']
    },
    'ground_water_recharge_ham_canals_nc': {
        'meaning': 'Recharge from canals in non-command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_canals_c', 'ground_water_recharge_ham_canals_pq', 'ground_water_recharge_ham_canals_total']
    },
    'ground_water_recharge_ham_canals_pq': {
        'meaning': 'Recharge from canals in poor quality area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_canals_c', 'ground_water_recharge_ham_canals_nc', 'ground_water_recharge_ham_canals_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_canals_total': {
        'meaning': 'Total recharge from canals (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1000 ham, Moderate: 1000-2000 ham, High: 2000-4000 ham, Very High: 4000-8000 ham, Extremely High: >8000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_canals_c', 'ground_water_recharge_ham_canals_nc', 'ground_water_recharge_ham_canals_pq']
    },
    'ground_water_recharge_ham_surface_water_irrigation_c': {
        'meaning': 'Recharge from surface water irrigation in command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_surface_water_irrigation_nc', 'ground_water_recharge_ham_surface_water_irrigation_pq', 'ground_water_recharge_ham_surface_water_irrigation_total']
    },
    'ground_water_recharge_ham_surface_water_irrigation_nc': {
        'meaning': 'Recharge from surface water irrigation in non-command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_surface_water_irrigation_c', 'ground_water_recharge_ham_surface_water_irrigation_pq', 'ground_water_recharge_ham_surface_water_irrigation_total']
    },
    'ground_water_recharge_ham_surface_water_irrigation_pq': {
        'meaning': 'Recharge from surface water irrigation in poor quality area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_surface_water_irrigation_c', 'ground_water_recharge_ham_surface_water_irrigation_nc', 'ground_water_recharge_ham_surface_water_irrigation_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_surface_water_irrigation_total': {
        'meaning': 'Total recharge from surface water irrigation (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1000 ham, Moderate: 1000-2000 ham, High: 2000-4000 ham, Very High: 4000-8000 ham, Extremely High: >8000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_surface_water_irrigation_c', 'ground_water_recharge_ham_surface_water_irrigation_nc', 'ground_water_recharge_ham_surface_water_irrigation_pq']
    },
    'ground_water_recharge_ham_ground_water_irrigation_c': {
        'meaning': 'Recharge from groundwater irrigation in command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_ground_water_irrigation_nc', 'ground_water_recharge_ham_ground_water_irrigation_pq', 'ground_water_recharge_ham_ground_water_irrigation_total']
    },
    'ground_water_recharge_ham_ground_water_irrigation_nc': {
        'meaning': 'Recharge from groundwater irrigation in non-command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_ground_water_irrigation_c', 'ground_water_recharge_ham_ground_water_irrigation_pq', 'ground_water_recharge_ham_ground_water_irrigation_total']
    },
    'ground_water_recharge_ham_ground_water_irrigation_pq': {
        'meaning': 'Recharge from groundwater irrigation in poor quality area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-5000 ham, Extremely High: >5000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_ground_water_irrigation_c', 'ground_water_recharge_ham_ground_water_irrigation_nc', 'ground_water_recharge_ham_ground_water_irrigation_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_ground_water_irrigation_total': {
        'meaning': 'Total recharge from groundwater irrigation (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1000 ham, Moderate: 1000-2000 ham, High: 2000-4000 ham, Very High: 4000-8000 ham, Extremely High: >8000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_ground_water_irrigation_c', 'ground_water_recharge_ham_ground_water_irrigation_nc', 'ground_water_recharge_ham_ground_water_irrigation_pq']
    },
    'ground_water_recharge_ham_tanks_and_ponds_c': {
        'meaning': 'Recharge from tanks and ponds in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_tanks_and_ponds_nc', 'ground_water_recharge_ham_tanks_and_ponds_pq', 'ground_water_recharge_ham_tanks_and_ponds_total']
    },
    'ground_water_recharge_ham_tanks_and_ponds_nc': {
        'meaning': 'Recharge from tanks and ponds in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_tanks_and_ponds_c', 'ground_water_recharge_ham_tanks_and_ponds_pq', 'ground_water_recharge_ham_tanks_and_ponds_total']
    },
    'ground_water_recharge_ham_tanks_and_ponds_pq': {
        'meaning': 'Recharge from tanks and ponds in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_tanks_and_ponds_c', 'ground_water_recharge_ham_tanks_and_ponds_nc', 'ground_water_recharge_ham_tanks_and_ponds_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_tanks_and_ponds_total': {
        'meaning': 'Total recharge from tanks and ponds (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_tanks_and_ponds_c', 'ground_water_recharge_ham_tanks_and_ponds_nc', 'ground_water_recharge_ham_tanks_and_ponds_pq']
    },
    'ground_water_recharge_ham_water_conservation_structure_c': {
        'meaning': 'Recharge from water conservation structures in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_water_conservation_structure_nc', 'ground_water_recharge_ham_water_conservation_structure_pq', 'ground_water_recharge_ham_water_conservation_structure_total']
    },
    'ground_water_recharge_ham_water_conservation_structure_nc': {
        'meaning': 'Recharge from water conservation structures in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_water_conservation_structure_c', 'ground_water_recharge_ham_water_conservation_structure_pq', 'ground_water_recharge_ham_water_conservation_structure_total']
    },
    'ground_water_recharge_ham_water_conservation_structure_pq': {
        'meaning': 'Recharge from water conservation structures in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_water_conservation_structure_c', 'ground_water_recharge_ham_water_conservation_structure_nc', 'ground_water_recharge_ham_water_conservation_structure_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_water_conservation_structure_total': {
        'meaning': 'Total recharge from water conservation structures (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_water_conservation_structure_c', 'ground_water_recharge_ham_water_conservation_structure_nc', 'ground_water_recharge_ham_water_conservation_structure_pq']
    },
    'ground_water_recharge_ham_pipelines_c': {
        'meaning': 'Recharge from pipelines in command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>1000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_pipelines_nc', 'ground_water_recharge_ham_pipelines_pq', 'ground_water_recharge_ham_pipelines_total']
    },
    'ground_water_recharge_ham_pipelines_nc': {
        'meaning': 'Recharge from pipelines in non-command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>1000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_pipelines_c', 'ground_water_recharge_ham_pipelines_pq', 'ground_water_recharge_ham_pipelines_total']
    },
    'ground_water_recharge_ham_pipelines_pq': {
        'meaning': 'Recharge from pipelines in poor quality area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Saline: If >1000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<50 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_pipelines_c', 'ground_water_recharge_ham_pipelines_nc', 'ground_water_recharge_ham_pipelines_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_pipelines_total': {
        'meaning': 'Total recharge from pipelines (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_pipelines_c', 'ground_water_recharge_ham_pipelines_nc', 'ground_water_recharge_ham_pipelines_pq']
    },
    'ground_water_recharge_ham_sewages_and_flash_flood_channels_c': {
        'meaning': 'Recharge from sewage/flash floods in command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>1000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_sewages_and_flash_flood_channels_nc', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_pq', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_total']
    },
    'ground_water_recharge_ham_sewages_and_flash_flood_channels_nc': {
        'meaning': 'Recharge from sewage/flash floods in non-command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>1000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_sewages_and_flash_flood_channels_c', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_pq', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_total']
    },
    'ground_water_recharge_ham_sewages_and_flash_flood_channels_pq': {
        'meaning': 'Recharge from sewage/flash floods in poor quality area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Saline: If >1000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<50 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_sewages_and_flash_flood_channels_c', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_nc', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_sewages_and_flash_flood_channels_total': {
        'meaning': 'Total recharge from sewage/flash floods (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_sewages_and_flash_flood_channels_c', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_nc', 'ground_water_recharge_ham_sewages_and_flash_flood_channels_pq']
    },
    'ground_water_recharge_ham_c': {
        'meaning': 'Total groundwater recharge in command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_nc', 'ground_water_recharge_ham_pq', 'ground_water_recharge_ham_total']
    },
    'ground_water_recharge_ham_nc': {
        'meaning': 'Total groundwater recharge in non-command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_c', 'ground_water_recharge_ham_pq', 'ground_water_recharge_ham_total']
    },
    'ground_water_recharge_ham_pq': {
        'meaning': 'Total groundwater recharge in poor quality area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_c', 'ground_water_recharge_ham_nc', 'ground_water_recharge_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_recharge_ham_total': {
        'meaning': 'Total groundwater recharge across all areas (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_recharge_ham_c', 'ground_water_recharge_ham_nc', 'ground_water_recharge_ham_pq']
    },
    'inflows_and_outflows_ham_base_flow_c': {
        'meaning': 'Base flow inflows in command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_base_flow_nc', 'inflows_and_outflows_ham_base_flow_pq', 'inflows_and_outflows_ham_base_flow_total']
    },
    'inflows_and_outflows_ham_base_flow_nc': {
        'meaning': 'Base flow inflows in non-command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_base_flow_c', 'inflows_and_outflows_ham_base_flow_pq', 'inflows_and_outflows_ham_base_flow_total']
    },
    'inflows_and_outflows_ham_base_flow_pq': {
        'meaning': 'Base flow inflows in poor quality area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_base_flow_c', 'inflows_and_outflows_ham_base_flow_nc', 'inflows_and_outflows_ham_base_flow_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_base_flow_total': {
        'meaning': 'Total base flow inflows (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1000 ham, Moderate: 1000-2000 ham, High: 2000-4000 ham, Very High: 4000-8000 ham, Extremely High: >8000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_base_flow_c', 'inflows_and_outflows_ham_base_flow_nc', 'inflows_and_outflows_ham_base_flow_pq']
    },
    'inflows_and_outflows_ham_stream_recharges_c': {
        'meaning': 'Recharge from streams in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_stream_recharges_nc', 'inflows_and_outflows_ham_stream_recharges_pq', 'inflows_and_outflows_ham_stream_recharges_total']
    },
    'inflows_and_outflows_ham_stream_recharges_nc': {
        'meaning': 'Recharge from streams in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_stream_recharges_c', 'inflows_and_outflows_ham_stream_recharges_pq', 'inflows_and_outflows_ham_stream_recharges_total']
    },
    'inflows_and_outflows_ham_stream_recharges_pq': {
        'meaning': 'Recharge from streams in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_stream_recharges_c', 'inflows_and_outflows_ham_stream_recharges_nc', 'inflows_and_outflows_ham_stream_recharges_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_stream_recharges_total': {
        'meaning': 'Total recharge from streams (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_stream_recharges_c', 'inflows_and_outflows_ham_stream_recharges_nc', 'inflows_and_outflows_ham_stream_recharges_pq']
    },
    'inflows_and_outflows_ham_lateral_flows_c': {
        'meaning': 'Lateral inflows in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_lateral_flows_nc', 'inflows_and_outflows_ham_lateral_flows_pq', 'inflows_and_outflows_ham_lateral_flows_total']
    },
    'inflows_and_outflows_ham_lateral_flows_nc': {
        'meaning': 'Lateral inflows in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_lateral_flows_c', 'inflows_and_outflows_ham_lateral_flows_pq', 'inflows_and_outflows_ham_lateral_flows_total']
    },
    'inflows_and_outflows_ham_lateral_flows_pq': {
        'meaning': 'Lateral inflows in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_lateral_flows_c', 'inflows_and_outflows_ham_lateral_flows_nc', 'inflows_and_outflows_ham_lateral_flows_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_lateral_flows_total': {
        'meaning': 'Total lateral inflows (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_lateral_flows_c', 'inflows_and_outflows_ham_lateral_flows_nc', 'inflows_and_outflows_ham_lateral_flows_pq']
    },
    'inflows_and_outflows_ham_vertical_flows_c': {
        'meaning': 'Vertical inflows in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_vertical_flows_nc', 'inflows_and_outflows_ham_vertical_flows_pq', 'inflows_and_outflows_ham_vertical_flows_total']
    },
    'inflows_and_outflows_ham_vertical_flows_nc': {
        'meaning': 'Vertical inflows in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_vertical_flows_c', 'inflows_and_outflows_ham_vertical_flows_pq', 'inflows_and_outflows_ham_vertical_flows_total']
    },
    'inflows_and_outflows_ham_vertical_flows_pq': {
        'meaning': 'Vertical inflows in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_vertical_flows_c', 'inflows_and_outflows_ham_vertical_flows_nc', 'inflows_and_outflows_ham_vertical_flows_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_vertical_flows_total': {
        'meaning': 'Total vertical inflows (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_vertical_flows_c', 'inflows_and_outflows_ham_vertical_flows_nc', 'inflows_and_outflows_ham_vertical_flows_pq']
    },
    'inflows_and_outflows_ham_evaporation_c': {
        'meaning': 'Evaporation loss in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evaporation_nc', 'inflows_and_outflows_ham_evaporation_pq', 'inflows_and_outflows_ham_evaporation_total']
    },
    'inflows_and_outflows_ham_evaporation_nc': {
        'meaning': 'Evaporation loss in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evaporation_c', 'inflows_and_outflows_ham_evaporation_pq', 'inflows_and_outflows_ham_evaporation_total']
    },
    'inflows_and_outflows_ham_evaporation_pq': {
        'meaning': 'Evaporation loss in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >1200 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evaporation_c', 'inflows_and_outflows_ham_evaporation_nc', 'inflows_and_outflows_ham_evaporation_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_evaporation_total': {
        'meaning': 'Total evaporation loss (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2400 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evaporation_c', 'inflows_and_outflows_ham_evaporation_nc', 'inflows_and_outflows_ham_evaporation_pq']
    },
    'inflows_and_outflows_ham_transpiration_c': {
        'meaning': 'Transpiration loss in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_transpiration_nc', 'inflows_and_outflows_ham_transpiration_pq', 'inflows_and_outflows_ham_transpiration_total']
    },
    'inflows_and_outflows_ham_transpiration_nc': {
        'meaning': 'Transpiration loss in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_transpiration_c', 'inflows_and_outflows_ham_transpiration_pq', 'inflows_and_outflows_ham_transpiration_total']
    },
    'inflows_and_outflows_ham_transpiration_pq': {
        'meaning': 'Transpiration loss in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >1200 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_transpiration_c', 'inflows_and_outflows_ham_transpiration_nc', 'inflows_and_outflows_ham_transpiration_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_transpiration_total': {
        'meaning': 'Total transpiration loss (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2400 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_transpiration_c', 'inflows_and_outflows_ham_transpiration_nc', 'inflows_and_outflows_ham_transpiration_pq']
    },
    'inflows_and_outflows_ham_evapotranspiration_c': {
        'meaning': 'Evapotranspiration loss in command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evapotranspiration_nc', 'inflows_and_outflows_ham_evapotranspiration_pq', 'inflows_and_outflows_ham_evapotranspiration_total']
    },
    'inflows_and_outflows_ham_evapotranspiration_nc': {
        'meaning': 'Evapotranspiration loss in non-command area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evapotranspiration_c', 'inflows_and_outflows_ham_evapotranspiration_pq', 'inflows_and_outflows_ham_evapotranspiration_total']
    },
    'inflows_and_outflows_ham_evapotranspiration_pq': {
        'meaning': 'Evapotranspiration loss in poor quality area (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Saline: If >2000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<200 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evapotranspiration_c', 'inflows_and_outflows_ham_evapotranspiration_nc', 'inflows_and_outflows_ham_evapotranspiration_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_evapotranspiration_total': {
        'meaning': 'Total evapotranspiration loss (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1000 ham, Moderate: 1000-2000 ham, High: 2000-4000 ham, Very High: 4000-8000 ham, Extremely High: >8000 ham.',
        'classification': 'Not a primary classifier. High (>4000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_evapotranspiration_c', 'inflows_and_outflows_ham_evapotranspiration_nc', 'inflows_and_outflows_ham_evapotranspiration_pq']
    },
    'inflows_and_outflows_ham_c': {
        'meaning': 'Total inflows and outflows in command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<500 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_nc', 'inflows_and_outflows_ham_pq', 'inflows_and_outflows_ham_total']
    },
    'inflows_and_outflows_ham_nc': {
        'meaning': 'Total inflows and outflows in non-command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<500 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_c', 'inflows_and_outflows_ham_pq', 'inflows_and_outflows_ham_total']
    },
    'inflows_and_outflows_ham_pq': {
        'meaning': 'Total inflows and outflows in poor quality area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<500 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_c', 'inflows_and_outflows_ham_nc', 'inflows_and_outflows_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'inflows_and_outflows_ham_total': {
        'meaning': 'Total inflows and outflows across all areas (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-12000 ham, Very High: 12000-20000 ham, Extremely High: >20000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['inflows_and_outflows_ham_c', 'inflows_and_outflows_ham_nc', 'inflows_and_outflows_ham_pq']
    },
    'annual_ground_water_recharge_ham_c': {
        'meaning': 'Annual groundwater recharge in command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_ground_water_recharge_ham_nc', 'annual_ground_water_recharge_ham_pq', 'annual_ground_water_recharge_ham_total']
    },
    'annual_ground_water_recharge_ham_nc': {
        'meaning': 'Annual groundwater recharge in non-command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_ground_water_recharge_ham_c', 'annual_ground_water_recharge_ham_pq', 'annual_ground_water_recharge_ham_total']
    },
    'annual_ground_water_recharge_ham_pq': {
        'meaning': 'Annual groundwater recharge in poor quality area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_ground_water_recharge_ham_c', 'annual_ground_water_recharge_ham_nc', 'annual_ground_water_recharge_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'annual_ground_water_recharge_ham_total': {
        'meaning': 'Total annual groundwater recharge (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_ground_water_recharge_ham_c', 'annual_ground_water_recharge_ham_nc', 'annual_ground_water_recharge_ham_pq']
    },
    'environmental_flows_ham_c': {
        'meaning': 'Environmental flow requirement in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['environmental_flows_ham_nc', 'environmental_flows_ham_pq', 'environmental_flows_ham_total']
    },
    'environmental_flows_ham_nc': {
        'meaning': 'Environmental flow requirement in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<100 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['environmental_flows_ham_c', 'environmental_flows_ham_pq', 'environmental_flows_ham_total']
    },
    'environmental_flows_ham_pq': {
        'meaning': 'Environmental flow requirement in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >1200 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['environmental_flows_ham_c', 'environmental_flows_ham_nc', 'environmental_flows_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'environmental_flows_ham_total': {
        'meaning': 'Total environmental flow requirement (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<200 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['environmental_flows_ham_c', 'environmental_flows_ham_nc', 'environmental_flows_ham_pq']
    },
    'annual_extractable_ground_water_resource_ham_c': {
        'meaning': 'Annual extractable groundwater resource in command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_extractable_ground_water_resource_ham_nc', 'annual_extractable_ground_water_resource_ham_pq', 'annual_extractable_ground_water_resource_ham_total']
    },
    'annual_extractable_ground_water_resource_ham_nc': {
        'meaning': 'Annual extractable groundwater resource in non-command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_extractable_ground_water_resource_ham_c', 'annual_extractable_ground_water_resource_ham_pq', 'annual_extractable_ground_water_resource_ham_total']
    },
    'annual_extractable_ground_water_resource_ham_pq': {
        'meaning': 'Annual extractable groundwater resource in poor quality area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_extractable_ground_water_resource_ham_c', 'annual_extractable_ground_water_resource_ham_nc', 'annual_extractable_ground_water_resource_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'annual_extractable_ground_water_resource_ham_total': {
        'meaning': 'Total annual extractable groundwater resource (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['annual_extractable_ground_water_resource_ham_c', 'annual_extractable_ground_water_resource_ham_nc', 'annual_extractable_ground_water_resource_ham_pq']
    },
    'ground_water_extraction_for_all_uses_ham_domestic_c': {
        'meaning': 'Groundwater extraction for domestic use in command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_domestic_nc', 'ground_water_extraction_for_all_uses_ham_domestic_pq', 'ground_water_extraction_for_all_uses_ham_domestic_total']
    },
    'ground_water_extraction_for_all_uses_ham_domestic_nc': {
        'meaning': 'Groundwater extraction for domestic use in non-command area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_domestic_c', 'ground_water_extraction_for_all_uses_ham_domestic_pq', 'ground_water_extraction_for_all_uses_ham_domestic_total']
    },
    'ground_water_extraction_for_all_uses_ham_domestic_pq': {
        'meaning': 'Groundwater extraction for domestic use in poor quality area (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >1200 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_domestic_c', 'ground_water_extraction_for_all_uses_ham_domestic_nc', 'ground_water_extraction_for_all_uses_ham_domestic_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_extraction_for_all_uses_ham_domestic_total': {
        'meaning': 'Total groundwater extraction for domestic use (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_domestic_c', 'ground_water_extraction_for_all_uses_ham_domestic_nc', 'ground_water_extraction_for_all_uses_ham_domestic_pq']
    },
    'ground_water_extraction_for_all_uses_ham_industrial_c': {
        'meaning': 'Groundwater extraction for industrial use in command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>600 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_industrial_nc', 'ground_water_extraction_for_all_uses_ham_industrial_pq', 'ground_water_extraction_for_all_uses_ham_industrial_total']
    },
    'ground_water_extraction_for_all_uses_ham_industrial_nc': {
        'meaning': 'Groundwater extraction for industrial use in non-command area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Not a primary classifier. High (>600 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<50 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_industrial_c', 'ground_water_extraction_for_all_uses_ham_industrial_pq', 'ground_water_extraction_for_all_uses_ham_industrial_total']
    },
    'ground_water_extraction_for_all_uses_ham_industrial_pq': {
        'meaning': 'Groundwater extraction for industrial use in poor quality area (ham).',
        'categorization': 'Very Low: <50 ham, Low: 50-150 ham, Moderate: 150-300 ham, High: 300-600 ham, Very High: 600-1000 ham, Extremely High: >1000 ham.',
        'classification': 'Saline: If >600 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<50 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_industrial_c', 'ground_water_extraction_for_all_uses_ham_industrial_nc', 'ground_water_extraction_for_all_uses_ham_industrial_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_extraction_for_all_uses_ham_industrial_total': {
        'meaning': 'Total groundwater extraction for industrial use (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_industrial_c', 'ground_water_extraction_for_all_uses_ham_industrial_nc', 'ground_water_extraction_for_all_uses_ham_industrial_pq']
    },
    'ground_water_extraction_for_all_uses_ham_irrigation_c': {
        'meaning': 'Groundwater extraction for irrigation in command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>6000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_irrigation_nc', 'ground_water_extraction_for_all_uses_ham_irrigation_pq', 'ground_water_extraction_for_all_uses_ham_irrigation_total']
    },
    'ground_water_extraction_for_all_uses_ham_irrigation_nc': {
        'meaning': 'Groundwater extraction for irrigation in non-command area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>6000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_irrigation_c', 'ground_water_extraction_for_all_uses_ham_irrigation_pq', 'ground_water_extraction_for_all_uses_ham_irrigation_total']
    },
    'ground_water_extraction_for_all_uses_ham_irrigation_pq': {
        'meaning': 'Groundwater extraction for irrigation in poor quality area (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Saline: If >6000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<500 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_irrigation_c', 'ground_water_extraction_for_all_uses_ham_irrigation_nc', 'ground_water_extraction_for_all_uses_ham_irrigation_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_extraction_for_all_uses_ham_irrigation_total': {
        'meaning': 'Total groundwater extraction for irrigation (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-12000 ham, Very High: 12000-20000 ham, Extremely High: >20000 ham.',
        'classification': 'Not a primary classifier. High (>12000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_irrigation_c', 'ground_water_extraction_for_all_uses_ham_irrigation_nc', 'ground_water_extraction_for_all_uses_ham_irrigation_pq']
    },
    'ground_water_extraction_for_all_uses_ham_c': {
        'meaning': 'Total groundwater extraction for all uses in command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_nc', 'ground_water_extraction_for_all_uses_ham_pq', 'ground_water_extraction_for_all_uses_ham_total']
    },
    'ground_water_extraction_for_all_uses_ham_nc': {
        'meaning': 'Total groundwater extraction for all uses in non-command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_c', 'ground_water_extraction_for_all_uses_ham_pq', 'ground_water_extraction_for_all_uses_ham_total']
    },
    'ground_water_extraction_for_all_uses_ham_pq': {
        'meaning': 'Total groundwater extraction for all uses in poor quality area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_c', 'ground_water_extraction_for_all_uses_ham_nc', 'ground_water_extraction_for_all_uses_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'ground_water_extraction_for_all_uses_ham_total': {
        'meaning': 'Total groundwater extraction for all uses (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<2000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['ground_water_extraction_for_all_uses_ham_c', 'ground_water_extraction_for_all_uses_ham_nc', 'ground_water_extraction_for_all_uses_ham_pq']
    },
    'stage_of_ground_water_extraction_percent_c': {
        'meaning': 'Stage of groundwater extraction in command area (%).',
        'categorization': 'Safe: ≤70%, Semi Critical: 70-90%, Critical: 90-100%, Over Exploited: >100%.',
        'classification': 'Primary classifier for groundwater status in command area. Safe (≤70%), Semi Critical (70-90%), Critical (90-100%), Over Exploited (>100%). If missing, supports No Data.',
        'related_columns': ['stage_of_ground_water_extraction_percent_nc', 'stage_of_ground_water_extraction_percent_pq', 'stage_of_ground_water_extraction_percent_total']
    },
    'stage_of_ground_water_extraction_percent_nc': {
        'meaning': 'Stage of groundwater extraction in non-command area (%).',
        'categorization': 'Safe: ≤70%, Semi Critical: 70-90%, Critical: 90-100%, Over Exploited: >100%.',
        'classification': 'Primary classifier for groundwater status in non-command area. Safe (≤70%), Semi Critical (70-90%), Critical (90-100%), Over Exploited (>100%). If missing, supports No Data.',
        'related_columns': ['stage_of_ground_water_extraction_percent_c', 'stage_of_ground_water_extraction_percent_pq', 'stage_of_ground_water_extraction_percent_total']
    },
    'stage_of_ground_water_extraction_percent_pq': {
        'meaning': 'Stage of groundwater extraction in poor quality area (%).',
        'categorization': 'Safe: ≤70%, Semi Critical: 70-90%, Critical: 90-100%, Over Exploited: >100%.',
        'classification': 'Primary classifier for groundwater status in poor quality area. Saline: If >70% and quality_tagging_major_parameter_present_pq indicates “Salinity”. Otherwise, Safe (≤70%), Semi Critical (70-90%), Critical (90-100%), Over Exploited (>100%). If missing, supports No Data.',
        'related_columns': ['stage_of_ground_water_extraction_percent_c', 'stage_of_ground_water_extraction_percent_nc', 'stage_of_ground_water_extraction_percent_total', 'quality_tagging_major_parameter_present_pq']
    },
    'stage_of_ground_water_extraction_percent_total': {
        'meaning': 'Total stage of groundwater extraction (%).',
        'categorization': 'Safe: ≤70%, Semi Critical: 70-90%, Critical: 90-100%, Over Exploited: >100%.',
        'classification': 'Primary classifier for overall groundwater status. Safe (≤70%), Semi Critical (70-90%), Critical (90-100%), Over Exploited (>100%). If missing, supports No Data. Hilly Area if total_geographical_area_ha_hilly_area >30% of total_geographical_area_ha_total. Saline if quality_tagging_major_parameter_present_pq indicates “Salinity”.',
        'related_columns': ['stage_of_ground_water_extraction_percent_c', 'stage_of_ground_water_extraction_percent_nc', 'stage_of_ground_water_extraction_percent_pq', 'total_geographical_area_ha_hilly_area', 'quality_tagging_major_parameter_present_pq']
    },
    'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_c': {
        'meaning': 'Groundwater allocation for domestic use in command area for 2025 (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_c >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_nc', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_pq', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_total']
    },
    'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_nc': {
        'meaning': 'Groundwater allocation for domestic use in non-command area for 2025 (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Not a primary classifier. High (>1200 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_nc >90%; Very Low (<100 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_c', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_pq', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_total']
    },
    'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_pq': {
        'meaning': 'Groundwater allocation for domestic use in poor quality area for 2025 (ham).',
        'categorization': 'Very Low: <100 ham, Low: 100-300 ham, Moderate: 300-600 ham, High: 600-1200 ham, Very High: 1200-2000 ham, Extremely High: >2000 ham.',
        'classification': 'Saline: If >1200 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<100 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_c', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_nc', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_total': {
        'meaning': 'Total groundwater allocation for domestic use for 2025 (ham).',
        'categorization': 'Very Low: <200 ham, Low: 200-500 ham, Moderate: 500-1000 ham, High: 1000-2000 ham, Very High: 2000-4000 ham, Extremely High: >4000 ham.',
        'classification': 'Not a primary classifier. High (>2000 ham) may support Critical or Over Exploited if stage_of_ground_water_extraction_percent_total >90%; Very Low (<200 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_c', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_nc', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham_pq']
    },
    'net_annual_ground_water_availability_for_future_use_ham_c': {
        'meaning': 'Net annual groundwater availability for future use in command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_c ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['net_annual_ground_water_availability_for_future_use_ham_nc', 'net_annual_ground_water_availability_for_future_use_ham_pq', 'net_annual_ground_water_availability_for_future_use_ham_total']
    },
    'net_annual_ground_water_availability_for_future_use_ham_nc': {
        'meaning': 'Net annual groundwater availability for future use in non-command area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_nc ≤70%; Very Low (<1000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['net_annual_ground_water_availability_for_future_use_ham_c', 'net_annual_ground_water_availability_for_future_use_ham_pq', 'net_annual_ground_water_availability_for_future_use_ham_total']
    },
    'net_annual_ground_water_availability_for_future_use_ham_pq': {
        'meaning': 'Net annual groundwater availability for future use in poor quality area (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['net_annual_ground_water_availability_for_future_use_ham_c', 'net_annual_ground_water_availability_for_future_use_ham_nc', 'net_annual_ground_water_availability_for_future_use_ham_total', 'quality_tagging_major_parameter_present_pq']
    },
    'net_annual_ground_water_availability_for_future_use_ham_total': {
        'meaning': 'Total net annual groundwater availability for future use (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support Hilly Area or No Data. If missing or zero, supports No Data.',
        'related_columns': ['net_annual_ground_water_availability_for_future_use_ham_c', 'net_annual_ground_water_availability_for_future_use_ham_nc', 'net_annual_ground_water_availability_for_future_use_ham_pq']
    },
    'quality_tagging_major_parameter_present_c': {
        'meaning': 'Major water quality parameter present in command area.',
        'categorization': 'No categorization (categorical: e.g., Salinity, Fluoride, Nitrate, etc.).',
        'classification': 'Saline: If “Salinity” is present, supports Saline classification regardless of stage_of_ground_water_extraction_percent_c. Other parameters may influence usability but not primary classification. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_nc', 'quality_tagging_major_parameter_present_pq', 'quality_tagging_other_parameters_present_c']
    },
    'quality_tagging_major_parameter_present_nc': {
        'meaning': 'Major water quality parameter present in non-command area.',
        'categorization': 'No categorization (categorical: e.g., Salinity, Fluoride, Nitrate, etc.).',
        'classification': 'Saline: If “Salinity” is present, supports Saline classification regardless of stage_of_ground_water_extraction_percent_nc. Other parameters may influence usability but not primary classification. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_c', 'quality_tagging_major_parameter_present_pq', 'quality_tagging_other_parameters_present_nc']
    },
    'quality_tagging_major_parameter_present_pq': {
        'meaning': 'Major water quality parameter present in poor quality area.',
        'categorization': 'No categorization (categorical: e.g., Salinity, Fluoride, Nitrate, etc.).',
        'classification': 'Saline: If “Salinity” is present, supports Saline classification regardless of stage_of_ground_water_extraction_percent_pq. Other parameters may influence usability but not primary classification. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_c', 'quality_tagging_major_parameter_present_nc', 'quality_tagging_other_parameters_present_pq']
    },
    'quality_tagging_other_parameters_present_c': {
        'meaning': 'Other water quality parameters present in command area.',
        'categorization': 'No categorization (categorical: e.g., Arsenic, Heavy Metals, etc.).',
        'classification': 'Not a primary classifier. May influence usability but does not directly affect Safe, Semi Critical, Critical, or Over Exploited classification. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_c', 'quality_tagging_other_parameters_present_nc', 'quality_tagging_other_parameters_present_pq']
    },
    'quality_tagging_other_parameters_present_nc': {
        'meaning': 'Other water quality parameters present in non-command area.',
        'categorization': 'No categorization (categorical: e.g., Arsenic, Heavy Metals, etc.).',
        'classification': 'Not a primary classifier. May influence usability but does not directly affect Safe, Semi Critical, Critical, or Over Exploited classification. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_nc', 'quality_tagging_other_parameters_present_c', 'quality_tagging_other_parameters_present_pq']
    },
    'quality_tagging_other_parameters_present_pq': {
        'meaning': 'Other water quality parameters present in poor quality area.',
        'categorization': 'No categorization (categorical: e.g., Arsenic, Heavy Metals, etc.).',
        'classification': 'Not a primary classifier. May influence usability but does not directly affect Saline classification unless quality_tagging_major_parameter_present_pq indicates “Salinity”. If missing, supports No Data.',
        'related_columns': ['quality_tagging_major_parameter_present_pq', 'quality_tagging_other_parameters_present_c', 'quality_tagging_other_parameters_present_nc']
    },
    'additional_potential_resources_under_specific_conditions_ham_waterlogged_and_shallow_water_table': {
        'meaning': 'Additional potential groundwater resources in waterlogged and shallow water table areas (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>6000 ham) may support Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['additional_potential_resources_under_specific_conditions_ham_flood_prone', 'additional_potential_resources_under_specific_conditions_ham_spring_discharge']
    },
    'additional_potential_resources_under_specific_conditions_ham_flood_prone': {
        'meaning': 'Additional potential groundwater resources in flood-prone areas (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>6000 ham) may support Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['additional_potential_resources_under_specific_conditions_ham_waterlogged_and_shallow_water_table', 'additional_potential_resources_under_specific_conditions_ham_spring_discharge']
    },
    'additional_potential_resources_under_specific_conditions_ham_spring_discharge': {
        'meaning': 'Additional potential groundwater resources from spring discharge (ham).',
        'categorization': 'Very Low: <500 ham, Low: 500-1500 ham, Moderate: 1500-3000 ham, High: 3000-6000 ham, Very High: 6000-10000 ham, Extremely High: >10000 ham.',
        'classification': 'Not a primary classifier. High (>6000 ham) may support Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<500 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['additional_potential_resources_under_specific_conditions_ham_waterlogged_and_shallow_water_table', 'additional_potential_resources_under_specific_conditions_ham_flood_prone']
    },
    'coastal_areas_c': {
        'meaning': 'Coastal area in command area (hectares).',
        'categorization': 'Very Low: <1000 ha, Low: 1000-3000 ha, Moderate: 3000-6000 ha, High: 6000-10000 ha, Very High: 10000-15000 ha, Extremely High: >15000 ha.',
        'classification': 'Saline: If >10000 ha and quality_tagging_major_parameter_present_c indicates “Salinity”. Very Low (<1000 ha) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['coastal_areas_nc', 'coastal_areas_pq', 'coastal_areas_total', 'quality_tagging_major_parameter_present_c']
    },
    'coastal_areas_nc': {
        'meaning': 'Coastal area in non-command area (hectares).',
        'categorization': 'Very Low: <1000 ha, Low: 1000-3000 ha, Moderate: 3000-6000 ha, High: 6000-10000 ha, Very High: 10000-15000 ha, Extremely High: >15000 ha.',
        'classification': 'Saline: If >10000 ha and quality_tagging_major_parameter_present_nc indicates “Salinity”. Very Low (<1000 ha) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['coastal_areas_c', 'coastal_areas_pq', 'coastal_areas_total', 'quality_tagging_major_parameter_present_nc']
    },
    'coastal_areas_pq': {
        'meaning': 'Coastal area in poor quality area (hectares).',
        'categorization': 'Very Low: <1000 ha, Low: 1000-3000 ha, Moderate: 3000-6000 ha, High: 6000-10000 ha, Very High: 10000-15000 ha, Extremely High: >15000 ha.',
        'classification': 'Saline: If >10000 ha and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ha) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['coastal_areas_c', 'coastal_areas_nc', 'coastal_areas_total', 'quality_tagging_major_parameter_present_pq']
    },
    'coastal_areas_total': {
        'meaning': 'Total coastal area (hectares).',
        'categorization': 'Very Low: <2000 ha, Low: 2000-5000 ha, Moderate: 5000-10000 ha, High: 10000-20000 ha, Very High: 20000-30000 ha, Extremely High: >30000 ha.',
        'classification': 'Saline: If >20000 ha and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<2000 ha) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['coastal_areas_c', 'coastal_areas_nc', 'coastal_areas_pq', 'quality_tagging_major_parameter_present_pq']
    },
    'in_storage_unconfined_ground_water_resources_ham_fresh': {
        'meaning': 'In-storage unconfined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_unconfined_ground_water_resources_ham_saline', 'total_ground_water_availability_in_unconfined_aquifier_ham_fresh']
    },
    'in_storage_unconfined_ground_water_resources_ham_saline': {
        'meaning': 'In-storage unconfined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_unconfined_ground_water_resources_ham_fresh', 'total_ground_water_availability_in_unconfined_aquifier_ham_saline']
    },
    'total_ground_water_availability_in_unconfined_aquifier_ham_fresh': {
        'meaning': 'Total groundwater availability in unconfined aquifer (fresh) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_unconfined_ground_water_resources_ham_fresh', 'total_ground_water_availability_in_unconfined_aquifier_ham_saline']
    },
    'total_ground_water_availability_in_unconfined_aquifier_ham_saline': {
        'meaning': 'Total groundwater availability in unconfined aquifer (saline) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Saline: If >20000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<2000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_unconfined_ground_water_resources_ham_saline', 'total_ground_water_availability_in_unconfined_aquifier_ham_fresh']
    },
    'dynamic_confined_ground_water_resources_ham_fresh': {
        'meaning': 'Dynamic confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_confined_ground_water_resources_ham_saline', 'total_confined_ground_water_resources_ham_fresh']
    },
    'dynamic_confined_ground_water_resources_ham_saline': {
        'meaning': 'Dynamic confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_confined_ground_water_resources_ham_fresh', 'total_confined_ground_water_resources_ham_saline']
    },
    'in_storage_confined_ground_water_resources_ham_fresh': {
        'meaning': 'In-storage confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_confined_ground_water_resources_ham_saline', 'total_confined_ground_water_resources_ham_fresh']
    },
    'in_storage_confined_ground_water_resources_ham_saline': {
        'meaning': 'In-storage confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_confined_ground_water_resources_ham_fresh', 'total_confined_ground_water_resources_ham_saline']
    },
    'total_confined_ground_water_resources_ham_fresh': {
        'meaning': 'Total confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_confined_ground_water_resources_ham_fresh', 'in_storage_confined_ground_water_resources_ham_fresh', 'total_confined_ground_water_resources_ham_saline']
    },
    'total_confined_ground_water_resources_ham_saline': {
        'meaning': 'Total confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Saline: If >20000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<2000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_confined_ground_water_resources_ham_saline', 'in_storage_confined_ground_water_resources_ham_saline', 'total_confined_ground_water_resources_ham_fresh']
    },
    'dynamic_semi_confined_ground_water_resources_ham_fresh': {
        'meaning': 'Dynamic semi-confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_semi_confined_ground_water_resources_ham_saline', 'total_semi_confined_ground_water_resources_ham_fresh']
    },
    'dynamic_semi_confined_ground_water_resources_ham_saline': {
        'meaning': 'Dynamic semi-confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_semi_confined_ground_water_resources_ham_fresh', 'total_semi_confined_ground_water_resources_ham_saline']
    },
    'in_storage_semi_confined_ground_water_resources_ham_fresh': {
        'meaning': 'In-storage semi-confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Not a primary classifier. High (>10000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<1000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_semi_confined_ground_water_resources_ham_saline', 'total_semi_confined_ground_water_resources_ham_fresh']
    },
    'in_storage_semi_confined_ground_water_resources_ham_saline': {
        'meaning': 'In-storage semi-confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <1000 ham, Low: 1000-3000 ham, Moderate: 3000-6000 ham, High: 6000-10000 ham, Very High: 10000-15000 ham, Extremely High: >15000 ham.',
        'classification': 'Saline: If >10000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<1000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['in_storage_semi_confined_ground_water_resources_ham_fresh', 'total_semi_confined_ground_water_resources_ham_saline']
    },
    'total_semi_confined_ground_water_resources_ham_fresh': {
        'meaning': 'Total semi-confined groundwater resources (fresh) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Not a primary classifier. High (>20000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<2000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_semi_confined_ground_water_resources_ham_fresh', 'in_storage_semi_confined_ground_water_resources_ham_fresh', 'total_semi_confined_ground_water_resources_ham_saline']
    },
    'total_semi_confined_ground_water_resources_ham_saline': {
        'meaning': 'Total semi-confined groundwater resources (saline) (ham).',
        'categorization': 'Very Low: <2000 ham, Low: 2000-5000 ham, Moderate: 5000-10000 ham, High: 10000-20000 ham, Very High: 20000-30000 ham, Extremely High: >30000 ham.',
        'classification': 'Saline: If >20000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<2000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['dynamic_semi_confined_ground_water_resources_ham_saline', 'in_storage_semi_confined_ground_water_resources_ham_saline', 'total_semi_confined_ground_water_resources_ham_fresh']
    },
    'total_ground_water_availability_in_the_area_ham_fresh': {
        'meaning': 'Total groundwater availability in the area (fresh) (ham).',
        'categorization': 'Very Low: <5000 ham, Low: 5000-10000 ham, Moderate: 10000-20000 ham, High: 20000-40000 ham, Very High: 40000-60000 ham, Extremely High: >60000 ham.',
        'classification': 'Not a primary classifier. High (>40000 ham) supports Safe if stage_of_ground_water_extraction_percent_total ≤70%; Very Low (<5000 ham) may support No Data or Hilly Area. If missing or zero, supports No Data.',
        'related_columns': ['total_ground_water_availability_in_the_area_ham_saline', 'total_ground_water_availability_in_unconfined_aquifier_ham_fresh', 'total_confined_ground_water_resources_ham_fresh', 'total_semi_confined_ground_water_resources_ham_fresh']
    },
    'total_ground_water_availability_in_the_area_ham_saline': {
        'meaning': 'Total groundwater availability in the area (saline) (ham).',
        'categorization': 'Very Low: <5000 ham, Low: 5000-10000 ham, Moderate: 10000-20000 ham, High: 20000-40000 ham, Very High: 40000-60000 ham, Extremely High: >60000 ham.',
        'classification': 'Saline: If >40000 ham and quality_tagging_major_parameter_present_pq indicates “Salinity”. Very Low (<5000 ham) may support No Data. If missing or zero, supports No Data.',
        'related_columns': ['total_ground_water_availability_in_the_area_ham_fresh', 'total_ground_water_availability_in_unconfined_aquifier_ham_saline', 'total_confined_ground_water_resources_ham_saline', 'total_semi_confined_ground_water_resources_ham_saline']
    }
}

# Log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    total_mem = psutil.virtual_memory().total / (1024 ** 3)  # Total RAM in GB
    available_mem = psutil.virtual_memory().available / (1024 ** 3)  # Available RAM in GB
    used_mem = mem_info.rss / (1024 ** 3)  # Used by process in GB
    print(f"Memory Usage: Total={total_mem:.2f} GB, Available={available_mem:.2f} GB, Used by process={used_mem:.2f} GB")

# Cleanup stray Uvicorn processes
def cleanup_uvicorn():
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'].lower() in ['uvicorn', 'python.exe', 'pythonw.exe']:
            if 'http://0.0.0.0:5000' in ' '.join(proc.cmdline()):
                print(f"Terminating stray Uvicorn process (PID: {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()

# Load data from SQLite database
def load_data():
    db_path = r'C:\Users\dr-prashant\Downloads\sih\INGRES_db.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM my_table"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Columns in loaded DataFrame:", df.columns.tolist())
    if df.empty:
        raise ValueError("DataFrame is empty. Check the database or query.")
    return df

# Load data
try:
    df = load_data()
except Exception as e:
    print(f"Error loading data from database: {e}")
    traceback.print_exc()
    df = pd.DataFrame()
    exit()

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

# Initialize embedding model
try:
    embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    traceback.print_exc()
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

# Embeddings for Chroma
try:
    df['text'] = df.apply(lambda row: create_text_chunk(row, required_columns), axis=1)
    if not df['text'].str.strip().any():
        print("Error: No valid text chunks created. Check DataFrame content.")
        exit()
    texts = df['text'].tolist()
    embeddings = embed_model.encode(texts, show_progress_bar=True)
except Exception as e:
    print(f"Error creating embeddings: {e}")
    traceback.print_exc()
    embeddings = []
    exit()

# Chroma index
try:
    chroma_client = chromadb.PersistentClient(path="./gec_chroma_index")
    try:
        chroma_client.delete_collection("my_table")
    except:
        pass
    collection = chroma_client.create_collection(
        name="my_table",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-en-v1.5')
    )
    ids = [str(i) for i in range(len(df))]
    metadatas = [{'state': str(row.get('state', 'Unknown')), 'district': str(row.get('district', 'Unknown'))} for row in df.to_dict('records')]
    for i, meta in enumerate(metadatas):
        if any(v is None or v == '' for v in meta.values()):
            print(f"Warning: Invalid metadata at index {i}: {meta}. Replacing with 'Unknown'.")
            meta['state'] = meta.get('state', 'Unknown') or 'Unknown'
            meta['district'] = meta.get('district', 'Unknown') or 'Unknown'
    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=texts)
except Exception as e:
    print(f"Error initializing Chroma index: {e}")
    traceback.print_exc()
    exit()

# Ollama LLM
try:
    log_memory_usage()
    llm = OllamaLLM(model="gemma3:12b", base_url="http://localhost:11434")
except Exception as e:
    print(f"Error initializing gemma3:12b: {e}")
    traceback.print_exc()
    print("Trying fallback model gemma3:12b...")
    try:
        llm = OllamaLLM(model="gemma3:12b", base_url="http://localhost:11434")
    except Exception as e:
        print(f"Error initializing gemma3:12b: {e}")
        traceback.print_exc()
        llm = None
        print("Warning: LLM disabled. Using static responses.")

# Translation model
try:
    trans_model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(trans_model_name)
    trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_name)
except Exception as e:
    print(f"Error loading translation model: {e}")
    traceback.print_exc()
    tokenizer = None
    trans_model = None

def translate_text(text, src_lang, tgt_lang="en"):
    if tokenizer is None or trans_model is None:
        return text
    try:
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = trans_model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error translating text: {e}")
        traceback.print_exc()
        return text

# Retrieve function
def retrieve(query, k=5):
    try:
        df['state_normalized'] = df['state'].str.lower()
        df['district_normalized'] = df['district'].str.lower()
        filter = {}
        
        state_match = re.search(r'\bstate\s*:\s*([a-zA-Z\s]+)\b|\b([a-zA-Z\s]+)\s*state\b', query.lower())
        district_match = re.search(r'\bdistrict\s*:\s*([a-zA-Z\s]+)\b|\b([a-zA-Z\s]+)\s*district\b', query.lower())
        
        if state_match:
            state = (state_match.group(1) or state_match.group(2)).strip().lower()
            if state in df['state_normalized'].values:
                filter['state'] = df[df['state_normalized'] == state]['state'].iloc[0]
        
        if district_match:
            district = (district_match.group(1) or district_match.group(2)).strip().lower()
            if district in df['district_normalized'].values:
                filter['district'] = df[df['district_normalized'] == district]['district'].iloc[0]
        
        relevant_columns = extract_relevant_columns(query)
        results = collection.query(query_texts=[query], n_results=k, where=filter if filter else None)
        ids = [int(id) for id in results['ids'][0]]
        relevant_df = df.iloc[ids]
        context = '\n'.join(relevant_df.apply(lambda row: create_text_chunk(row, relevant_columns), axis=1).tolist())
        return context, relevant_df
    except Exception as e:
        print(f"Error in retrieval: {e}")
        traceback.print_exc()
        return "", df.head(k)

# Static response fallback
def static_response(query, relevant_df):
    query_lower = query.lower()
    if relevant_df.empty:
        return "No data found for this query."
    
    relevant_columns = extract_relevant_columns(query)
    for col in relevant_columns:
        if col in relevant_df.columns:
            value = relevant_df.iloc[0].get(col, 'N/A')
            defn = COLUMN_DEFINITIONS.get(col, {})
            meaning = defn.get('meaning', 'No meaning available')
            return f"{col.replace('_', ' ').title()}: {value if value != 'N/A' else 'Data not available'} ({meaning})"
    return "Query not supported in fallback mode."

# Simplified prompt template
prompt = PromptTemplate(
    input_variables=["query", "context", "relevant_defs"],
    template="""Query: {query}

Data (column names and values for direct and related columns):
{context}

Column Definitions (meaning, categorization, classification):
{relevant_defs}

Instructions:
- Use the provided data and definitions to answer the query.
- For what-if scenarios, assume proportional relationships (e.g., 30% rainfall increase = 30% recharge increase) unless specified otherwise in the classification.
- Use numerical values with units (mm, ham, %) from the data.
- Apply categorization and classification to interpret values.
- If data is unavailable (N/A), state 'Data not available'.
- Be concise, factual, and respond in English."""
)

chain = LLMChain(llm=llm, prompt=prompt) if llm else None

def is_absurd(answer, query):
    if len(answer.strip()) < 10 or not re.match(r'^[a-zA-Z0-9\s,.%mmham()]+$', answer):
        return True
    if any(k in query.lower() for k in ['rainfall', 'recharge', 'extraction', 'availability', 'stage']) and not re.search(r'\d+\.?\d* (mm|ham|%)', answer):
        return True
    return False

# Visualization function
def visualize(query, relevant_df):
    query_lower = query.lower()
    if not any(keyword in query_lower for keyword in ['plot', 'chart', 'graph', 'visualize', 'map']):
        return ""
    
    plt.figure(figsize=(12, 8))
    try:
        if 'map' in query_lower:
            return "Map visualization not supported in console mode."
        
        elif 'pie' in query_lower and len(relevant_df) == 1:
            row = relevant_df.iloc[0]
            labels = ['Extraction', 'Recharge']
            sizes = [pd.to_numeric(row.get('ground_water_extraction_for_all_uses_ha_m_total', 0), errors='coerce'),
                     pd.to_numeric(row.get('ground_water_recharge_ham_total', 0), errors='coerce')]
            if all(s > 0 for s in sizes):
                plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                plt.title(f'Extraction vs Recharge for {row.get("district", "Unknown")}')
            else:
                return "Insufficient data for pie chart."
        
        elif 'bar' in query_lower or 'chart' in query_lower:
            relevant_columns = extract_relevant_columns(query)
            for col in relevant_columns:
                if col in relevant_df.columns and col in ['rainfall_mm_total', 'ground_water_recharge_ham_total', 
                                                         'ground_water_extraction_for_all_uses_ha_m_total', 
                                                         'total_ground_water_availability_in_the_area_ham', 
                                                         'stage_of_ground_water_extraction_total']:
                    relevant_df[col] = pd.to_numeric(relevant_df[col], errors='coerce')
                    if relevant_df[col].isna().all():
                        continue
                    unit = 'mm' if 'rainfall' in col else '%' if 'stage' in col else 'ham'
                    sns.barplot(data=relevant_df, x='district', y=col)
                    plt.title(f"{col.replace('_', ' ').title()} by District ({unit})")
                    break
            else:
                return "No valid numeric data for visualization."
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return "Visualization generated."
    except Exception as e:
        print(f"Error generating visualization: {e}")
        traceback.print_exc()
        return f"Error generating visualization: {e}"
    finally:
        plt.close()

# Chat loop with prompt logging
def chat():
    print("GEC RAG Chatbot Ready! Ask about groundwater data (e.g., status, recharge, extraction, availability, comparisons, what-if scenarios). Type 'quit' to exit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'quit':
            break
        
        lang = 'en'
        if not re.match(r'^[a-zA-Z0-9\s,.?]+$', query):
            try:
                lang = detect(query)
            except:
                lang = 'en'
        query_en = translate_text(query, lang) if lang != 'en' else query
        if lang != 'en':
            print(f"Translated query to EN: {query_en}")
        
        relevant_columns = extract_relevant_columns(query_en)
        relevant_defs = '\n'.join([
            f"{col}: Meaning: {COLUMN_DEFINITIONS[col]['meaning']}; Categorization: {COLUMN_DEFINITIONS[col]['categorization']}; Classification: {COLUMN_DEFINITIONS[col]['classification']}"
            for col in relevant_columns if col in COLUMN_DEFINITIONS
        ])
        
        context, relevant_df = retrieve(query_en)
        
        # Log the full prompt
        full_prompt = prompt.format(query=query_en, context=context, relevant_defs=relevant_defs)
        print("\n=== Full Prompt Sent to LLM ===")
        print(full_prompt)
        print("=== End of Prompt ===")
        
        if llm and chain:
            try:
                answer = chain.run(query=query_en, context=context, relevant_defs=relevant_defs)
                if is_absurd(answer, query_en):
                    direct_prompt = prompt.format(query=query_en, context=context, relevant_defs=relevant_defs)
                    try:
                        answer = ollama.generate(model='gemma3:12b', prompt=direct_prompt)['response']
                    except Exception as e:
                        print(f"Error in direct Ollama query: {e}")
                        traceback.print_exc()
                        answer = static_response(query_en, relevant_df)
            except Exception as e:
                print(f"Error querying LLM: {e}")
                traceback.print_exc()
                answer = static_response(query_en, relevant_df)
        else:
            answer = static_response(query_en, relevant_df)
        
        if lang != 'en':
            answer = translate_text(answer, "en", lang)
        
        viz_msg = ""
        if any(k in query_en.lower() for k in ['plot', 'chart', 'graph', 'visualize', 'map']):
            viz_msg = visualize(query_en, relevant_df)
        if viz_msg:
            answer += f"\n{viz_msg}"
        
        print(f"Bot: {answer}")

if __name__ == "__main__":
    try:
        cleanup_uvicorn()
        log_memory_usage()
        chat()
    except Exception as e:
        print(f"Error running chatbot: {e}")
        traceback.print_exc()