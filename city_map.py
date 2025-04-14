import pandas as pd
from rapidfuzz import process, fuzz

#Load and clean the reference dataset ===
ref_df = pd.read_csv('indian cities states.csv', header=1)  # header=1 skips first row if it's junk

# Clean and standardize column names
ref_df.columns = ref_df.columns.str.strip().str.lower()

# Debug: Print columns to confirm structure
print(" Cleaned reference columns:", ref_df.columns.tolist())

# Rename proper columns (match the real column names exactly!)
ref_df = ref_df.rename(columns={
    'city/town': 'city',
    'state/                                               union territory*': 'state'
})

# Keep only city and state
ref_df = ref_df[['city', 'state']]

# Normalize city and state values
ref_df['city'] = ref_df['city'].str.strip().str.lower()
ref_df['state'] = ref_df['state'].str.strip().str.lower()

# Load and clean your working dataset ===
city_df = pd.read_csv('insights_final_merged_data.csv')
city_df.columns = city_df.columns.str.strip().str.lower()

# Normalize city column
city_df['city'] = city_df['city'].str.strip().str.lower()

# === STEP 3: Manual city aliases ===
city_aliases = {
    'gurgaon': 'gurugram',
    'bombay': 'mumbai',
    'madras': 'chennai',
    'calcutta': 'kolkata',
    'trivandrum': 'thiruvananthapuram',
}

# Extract unique cities and apply aliases
unique_cities = city_df['city'].drop_duplicates()
unique_cities = unique_cities.apply(lambda c: city_aliases.get(c, c))

# Fuzzy match cities ===
ref_city_list = ref_df['city'].tolist()
matched_cities = []

for city in unique_cities:
    match, score, _ = process.extractOne(city, ref_city_list, scorer=fuzz.token_sort_ratio)
    if score >= 85:
        state = ref_df.loc[ref_df['city'] == match, 'state'].values[0]
        matched_cities.append({
            'city': city,
            'matched_city': match,
            'state': state
        })
    else:
        matched_cities.append({
            'city': city,
            'matched_city': None,
            'state': None
        })
# Fuzzy matching ===
# (existing matching loop here)

# Manual corrections ===
manual_fixes = {
    'mohali':           ('mohali', 'punjab'),
    'gurugram':         ('gurugram', 'haryana'),
    'navi':             ('navi mumbai', 'maharashtra'),
    'delhi':            ('delhi', 'delhi'),
    'mumbai':           ('mumbai', 'maharashtra'),
    'sangli':           ('sangli', 'maharashtra'),
    'panchkula':        ('panchkula', 'haryana'),
    'hyderabad':        ('hyderabad', 'telangana'),
    'howrah':           ('howrah', 'west bengal'),
    'ranga':            ('ranga reddy', 'telangana'),
}

for row in matched_cities:
    if not row['state'] or row['matched_city'] != row['city']:
        city_key = row['city'].lower()
        if city_key in manual_fixes:
            row['matched_city'], row['state'] = manual_fixes[city_key]

# Format and export ===
mapped_df = pd.DataFrame(matched_cities)

# Convert city/state names to title case for output
mapped_df['city'] = mapped_df['city'].str.title()
mapped_df['matched_city'] = mapped_df['matched_city'].str.title()
mapped_df['state'] = mapped_df['state'].str.title()

# Export to CSV
mapped_df.to_csv('Mapped_city.csv', index=False)
print(" Mapped_Cities_Fuzzy.csv created successfully!")
