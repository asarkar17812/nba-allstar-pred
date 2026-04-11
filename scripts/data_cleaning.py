import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

# df_data = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=1)
# df_team_abbr = pd.read_excel("/Users/ayushsarkar/nba-hof-pred/nba-hof-pred/source/NBA ALL STAR DATA.xlsx", sheet_name=7)
# print(df_data)

script_dir = Path(__file__).parent
data_path = script_dir / ".." / "source" / "uncleaned" / "NBA ALL STAR DATA.xlsx"

df_data = pd.read_excel(data_path, sheet_name=1)
df_team_abbr = pd.read_excel(data_path, sheet_name=7)

# print(df_data)
df_data.columns = df_data.columns.str.strip()
df_data = df_data.drop(['Games Started', 'Unnamed: 0'], axis=1)

def convert_height(height_str):
    try:
        if pd.isna(height_str) or not isinstance(height_str, str):
            return 0
        # This handles both "6-11" and "6'11"
        parts = height_str.replace("'", "-").split('-')
        
        feet = int(parts[0])
        inches = int(parts[1]) if len(parts) > 1 else 0
        
        return (feet * 12) + inches
    except:
        return 0

df_data['Height_Inches'] = df_data['Height'].apply(convert_height)
df_data = df_data.drop('Height', axis=1)

df_clean = df_data.dropna(subset=['Player'])

print(df_clean)
# confirms the rest of the nulls are within field goal % related stats
print(df_clean.isnull().sum()) 

# setting these to 0, players didn't shoot certain shots (can't divide by 0 --> just set to 0)
df_clean = df_clean.fillna(0)
# print(df_clean.isnull().sum()) # just checking

target = df_clean.pop('All Star')
# 2. Put it back in at the end
df_clean['All Star'] = target
print(df_clean)





