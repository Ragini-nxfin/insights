# car_utils.py
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df):
    """
    Preprocess the dataset by cleaning and transforming columns for numerical comparisons.
    Returns a new DataFrame with cleaned data.
    """
    df_copy = df.copy()

    def extract_first_number(text):
        if pd.isna(text):
            return 0
        text = str(text)
        match = re.search(r'\d+\.?\d*', text)
        return float(match.group()) if match else 0

    df_copy['Mileage (ARAI)'] = df_copy['Mileage (ARAI)'].apply(extract_first_number)
    df_copy['Seating Capacity'] = df_copy['Seating Capacity'].astype(str).str.replace(r'\s*person(s)?\s*', '', regex=True, case=False)
    df_copy['Seating Capacity'] = df_copy['Seating Capacity'].str.replace(r'\s*(seater(s)?|seat(s)?)\s*', '', regex=True, case=False)
    seating_map = {'5 & 6': 6, '7 & 8': 8, '7 & 9': 9}
    df_copy['Seating Capacity'] = df_copy['Seating Capacity'].replace(seating_map)
    df_copy['Seating Capacity'] = df_copy['Seating Capacity'].apply(extract_first_number)
    df_copy['NCAP Rating'] = df_copy['NCAP Rating'].apply(extract_first_number)
    df_copy['Airbags'] = df_copy['Airbags'].apply(extract_first_number)
    df_copy['Seat Belt Warning'] = df_copy['Seat Belt Warning'].str.lower().map({'yes': 1, 'no': 0})

    if 'Transmission' in df_copy.columns:
        df_copy['Transmission'] = df_copy['Transmission'].str.lower().map({'automatic': 1, 'manual': 0}).fillna(0)

    yes_no_cols = find_yes_no_columns(df_copy)
    df_copy = convert_yes_no_to_binary(df_copy, yes_no_cols)

    return df_copy

def find_yes_no_columns(df):
    yes_no_columns = []
    for column in df.columns:
        unique_vals = df[column].dropna().astype(str).str.lower().unique()
        if set(unique_vals).issubset({'yes', 'no'}) and len(unique_vals) > 0:
            yes_no_columns.append(column)
    return yes_no_columns

def convert_yes_no_to_binary(df, yes_no_columns):
    df_copy = df.copy()
    for column in yes_no_columns:
        if df_copy[column].dtype not in [int, float]:
            df_copy[column] = df_copy[column].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    return df_copy

def find_highest_features_same_variant(df, yes_no_columns):
    df_copy = df.copy()
    df_copy['Feature_score'] = df_copy[yes_no_columns].sum(axis=1)
    grouped = df_copy.groupby(['Make', 'Model', 'Variant', 'City'])
    result = grouped.apply(lambda x: x[x['Feature_score'] == x['Feature_score'].max()])
    result = result.reset_index(drop=True)
    result = result.sort_values(by='Feature_score', ascending=False)
    print("Feature Score value: ")
    print(result['Feature_score'].head(10))
    return result[['Make', 'Model', 'Variant', 'City', 'Feature_score'] + yes_no_columns]

def new_suggestion(listed_df, make, model, variant, city, selected_price, selected_distance, selected_age, price_range_percent=10):
    same_car_df = listed_df[(listed_df['Make'] == make) &
                            (listed_df['Model'] == model) &
                            (listed_df['Variant'] == variant)].copy()
    if same_car_df.empty:
        return same_car_df, 0, 0
    price_lower_bound = selected_price * (1 - price_range_percent / 100)
    price_upper_bound = selected_price * (1 + price_range_percent / 100)
    price_filtered_df = same_car_df[(same_car_df['Price_numeric'] >= price_lower_bound) &
                                    (same_car_df['Price_numeric'] <= price_upper_bound)]
    if price_filtered_df.empty:
        return price_filtered_df, price_lower_bound, price_upper_bound
    selected_car = price_filtered_df[(price_filtered_df['Price_numeric'] == selected_price) &
                                     (price_filtered_df['Distance_numeric'] == selected_distance) &
                                     (price_filtered_df['Age'] == selected_age)]
    if selected_car.empty:
        return selected_car, price_lower_bound, price_upper_bound
    selected_car = selected_car.iloc[0]
    better_conditions = (
        (price_filtered_df['Mileage (ARAI)'].astype(float) > float(selected_car['Mileage (ARAI)'])) |
        (price_filtered_df['NCAP Rating'].astype(float) > float(selected_car['NCAP Rating'])) |
        (price_filtered_df['Airbags'].astype(float) > float(selected_car['Airbags'])) |
        (price_filtered_df['Seating Capacity'].astype(float) > float(selected_car['Seating Capacity'])) |
        (price_filtered_df['Seat Belt Warning'].astype(float) > float(selected_car['Seat Belt Warning'])) |
        (price_filtered_df['Transmission'].astype(float) > float(selected_car['Transmission'])) |
        (price_filtered_df['Distance_numeric'] < selected_distance) |
        (price_filtered_df['Age'] < selected_age)
    )
    better_options_df = price_filtered_df[better_conditions]
    better_options_df = better_options_df.sort_values(by=['Distance_numeric', 'Age'])
    return better_options_df, price_lower_bound, price_upper_bound

def generate_insight_same_variant(selected_car, better_options_data):
    better_options_df, price_lower_bound, price_upper_bound = better_options_data
    if better_options_df.empty:
        return {'status': 'no_better_options', 'comparisons': [], 'best_option': None, 'selected_car': selected_car,
                'price_lower_bound': price_lower_bound, 'price_upper_bound': price_upper_bound}
    best_option = better_options_df.iloc[0]
    comparisons = []
    if best_option['Mileage (ARAI)'] > selected_car['Mileage (ARAI)']:
        comparisons.append(f"better mileage ({best_option['Mileage (ARAI)']} km/l vs {selected_car['Mileage (ARAI)']} km/l)")
    if best_option['NCAP Rating'] > selected_car['NCAP Rating']:
        comparisons.append(f"higher NCAP Rating ({best_option['NCAP Rating']} vs {selected_car['NCAP Rating']})")
    if best_option['Airbags'] > selected_car['Airbags']:
        comparisons.append(f"more airbags ({int(best_option['Airbags'])} vs {int(selected_car['Airbags'])})")
    if best_option['Seating Capacity'] > selected_car['Seating Capacity']:
        comparisons.append(f"higher seating capacity ({int(best_option['Seating Capacity'])} vs {int(selected_car['Seating Capacity'])})")
    if best_option['Seat Belt Warning'] > selected_car['Seat Belt Warning']:
        comparisons.append("a seat belt warning feature")
    if best_option['Transmission'] > selected_car['Transmission']:
        comparisons.append("an automatic transmission")
    if best_option['Distance_numeric'] < selected_car['Distance_numeric']:
        comparisons.append(f"lower mileage ({int(best_option['Distance_numeric'])} km vs {int(selected_car['Distance_numeric'])} km)")
    if best_option['Age'] < selected_car['Age']:
        comparisons.append(f"a newer age ({int(best_option['Age'])} years vs {int(selected_car['Age'])} years)")
    return {
        'status': 'success',
        'comparisons': comparisons,
        'best_option': best_option,
        'selected_car': selected_car,
        'price_lower_bound': price_lower_bound,
        'price_upper_bound': price_upper_bound
    }

def finalize_insight_with_feature_score(listed_df, insight_data, yes_no_cols):
    if insight_data['status'] == 'no_better_options':
        return "No better options found in the same price range for this model and variant in your city."
    best_option = insight_data['best_option']
    selected_car = insight_data['selected_car']
    comparisons = insight_data['comparisons']
    price_lower_bound = insight_data['price_lower_bound']
    price_upper_bound = insight_data['price_upper_bound']
    selected_car_row = listed_df[(listed_df['Make'] == selected_car['Make']) &
                                 (listed_df['Model'] == selected_car['Model']) &
                                 (listed_df['Variant'] == selected_car['Variant']) &
                                 (listed_df['Price_numeric'] == selected_car['Price_numeric']) &
                                 (listed_df['Distance_numeric'] == selected_car['Distance_numeric']) &
                                 (listed_df['Age'] == selected_car['Age'])]
    best_option_row = listed_df[(listed_df['Make'] == best_option['Make']) &
                                (listed_df['Model'] == best_option['Model']) &
                                (listed_df['Variant'] == best_option['Variant']) &
                                (listed_df['Price_numeric'] == best_option['Price_numeric']) &
                                (listed_df['Distance_numeric'] == best_option['Distance_numeric']) &
                                (listed_df['Age'] == best_option['Age'])]
    selected_car_feature_score = selected_car_row[yes_no_cols].sum(axis=1).iloc[0] if not selected_car_row.empty else 0
    best_option_feature_score = best_option_row[yes_no_cols].sum(axis=1).iloc[0] if not best_option_row.empty else 0
    if best_option_feature_score > selected_car_feature_score:
        comparisons.append(f"a higher feature score ({int(best_option_feature_score)} vs {int(selected_car_feature_score)})")
    if not comparisons:
        return "No significantly better options found based on the specified features."
    comparison_str = ", ".join(comparisons)
    insight = f"In the same price range, you can get a {selected_car['Make']} {selected_car['Model']} {selected_car['Variant']} " \
              f"in {selected_car['City']} with {comparison_str} " \
              f"for Rs {int(best_option['Price_numeric'])}. " \
              f"This is within the price range of Rs {int(price_lower_bound)} to Rs {int(price_upper_bound)}."
    return insight

def new_suggestion_different_variant(listed_df, make, model, city, selected_price, selected_distance, selected_age, price_range_percent=10):
    same_car_df = listed_df[(listed_df['Make'] == make) &
                            (listed_df['Model'] == model)].copy()
    if same_car_df.empty:
        return same_car_df, 0, 0
    price_lower_bound = selected_price * (1 - price_range_percent / 100)
    price_upper_bound = selected_price * (1 + price_range_percent / 100)
    price_filtered_df = same_car_df[(same_car_df['Price_numeric'] >= price_lower_bound) &
                                    (same_car_df['Price_numeric'] <= price_upper_bound)]
    if price_filtered_df.empty:
        return price_filtered_df, price_lower_bound, price_upper_bound
    selected_car = price_filtered_df[(price_filtered_df['Price_numeric'] == selected_price) &
                                     (price_filtered_df['Distance_numeric'] == selected_distance) &
                                     (price_filtered_df['Age'] == selected_age)]
    if selected_car.empty:
        return selected_car, price_lower_bound, price_upper_bound
    selected_car = selected_car.iloc[0]
    better_conditions = (
        (price_filtered_df['Mileage (ARAI)'].astype(float) > float(selected_car['Mileage (ARAI)'])) |
        (price_filtered_df['NCAP Rating'].astype(float) > float(selected_car['NCAP Rating'])) |
        (price_filtered_df['Airbags'].astype(float) > float(selected_car['Airbags'])) |
        (price_filtered_df['Seating Capacity'].astype(float) > float(selected_car['Seating Capacity'])) |
        (price_filtered_df['Seat Belt Warning'].astype(float) > float(selected_car['Seat Belt Warning'])) |
        (price_filtered_df['Transmission'].astype(float) > float(selected_car['Transmission'])) |
        (price_filtered_df['Distance_numeric'] < selected_distance) |
        (price_filtered_df['Age'] < selected_age)
    )
    better_options_df = price_filtered_df[better_conditions]
    better_options_df = better_options_df.sort_values(by=['Distance_numeric', 'Age'])
    return better_options_df, price_lower_bound, price_upper_bound

def generate_insight_different_variant(selected_car, better_options_data):
    better_options_df, price_lower_bound, price_upper_bound = better_options_data
    if better_options_df.empty:
        return {'status': 'no_better_options', 'comparisons': [], 'best_option': None, 'selected_car': selected_car,
                'price_lower_bound': price_lower_bound, 'price_upper_bound': price_upper_bound}
    best_option = better_options_df.iloc[0]
    comparisons = []
    if best_option['Mileage (ARAI)'] > selected_car['Mileage (ARAI)']:
        comparisons.append(f"better mileage ({best_option['Mileage (ARAI)']} km/l vs {selected_car['Mileage (ARAI)']} km/l)")
    if best_option['NCAP Rating'] > selected_car['NCAP Rating']:
        comparisons.append(f"higher NCAP Rating ({best_option['NCAP Rating']} vs {selected_car['NCAP Rating']})")
    if best_option['Airbags'] > selected_car['Airbags']:
        comparisons.append(f"more airbags ({int(best_option['Airbags'])} vs {int(selected_car['Airbags'])})")
    if best_option['Seating Capacity'] > selected_car['Seating Capacity']:
        comparisons.append(f"higher seating capacity ({int(best_option['Seating Capacity'])} vs {int(selected_car['Seating Capacity'])})")
    if best_option['Seat Belt Warning'] > selected_car['Seat Belt Warning']:
        comparisons.append("a seat belt warning feature")
    if best_option['Transmission'] > selected_car['Transmission']:
        comparisons.append("an automatic transmission")
    if best_option['Distance_numeric'] < selected_car['Distance_numeric']:
        comparisons.append(f"lower mileage ({int(best_option['Distance_numeric'])} km vs {int(selected_car['Distance_numeric'])} km)")
    if best_option['Age'] < selected_car['Age']:
        comparisons.append(f"a newer age ({int(best_option['Age'])} years vs {int(selected_car['Age'])} years)")
    return {
        'status': 'success',
        'comparisons': comparisons,
        'best_option': best_option,
        'selected_car': selected_car,
        'price_lower_bound': price_lower_bound,
        'price_upper_bound': price_upper_bound
    }

def finalize_insight_with_feature_score_different_variant(listed_df, insight_data, yes_no_cols):
    if insight_data['status'] == 'no_better_options':
        return "No better options found in the same price range for this model in your city with different variants."
    best_option = insight_data['best_option']
    selected_car = insight_data['selected_car']
    comparisons = insight_data['comparisons']
    price_lower_bound = insight_data['price_lower_bound']
    price_upper_bound = insight_data['price_upper_bound']
    listed_list_binary = convert_yes_no_to_binary(listed_df, yes_no_cols)
    selected_car_row = listed_list_binary[(listed_list_binary['Make'] == selected_car['Make']) &
                                         (listed_list_binary['Model'] == selected_car['Model']) &
                                         (listed_list_binary['Variant'] == selected_car['Variant']) &
                                         (listed_list_binary['Price_numeric'] == selected_car['Price_numeric']) &
                                         (listed_list_binary['Distance_numeric'] == selected_car['Distance_numeric']) &
                                         (listed_list_binary['Age'] == selected_car['Age'])]
    best_option_row = listed_list_binary[(listed_list_binary['Make'] == best_option['Make']) &
                                        (listed_list_binary['Model'] == best_option['Model']) &
                                        (listed_list_binary['Variant'] == best_option['Variant']) &
                                        (listed_list_binary['Price_numeric'] == best_option['Price_numeric']) &
                                        (listed_list_binary['Distance_numeric'] == best_option['Distance_numeric']) &
                                        (listed_list_binary['Age'] == best_option['Age'])]
    selected_car_feature_score = selected_car_row[yes_no_cols].sum(axis=1).iloc[0] if not selected_car_row.empty else 0
    best_option_feature_score = best_option_row[yes_no_cols].sum(axis=1).iloc[0] if not best_option_row.empty else 0
    if best_option_feature_score > selected_car_feature_score:
        comparisons.append(f"a higher feature score ({int(best_option_feature_score)} vs {int(selected_car_feature_score)})")
    if not comparisons:
        return "No significantly better options found based on the specified features with different variants."
    comparison_str = ", ".join(comparisons)
    insight = f"In the same price range, you can get a {selected_car['Make']} {selected_car['Model']} {best_option['Variant']} " \
              f"in {selected_car['City']} with {comparison_str} " \
              f"for Rs {int(best_option['Price_numeric'])}. " \
              f"This is within the price range of Rs {int(price_lower_bound)} to Rs {int(price_upper_bound)}."
    return insight

def new_suggestion_no_restrictions(listed_df, selected_price, selected_distance, selected_age, price_range_percent=10):
    price_lower_bound = selected_price * (1 - price_range_percent / 100)
    price_upper_bound = selected_price * (1 + price_range_percent / 100)
    price_filtered_df = listed_df[(listed_df['Price_numeric'] >= price_lower_bound) &
                                  (listed_df['Price_numeric'] <= price_upper_bound)].copy()
    if price_filtered_df.empty:
        return price_filtered_df, price_lower_bound, price_upper_bound
    selected_car = price_filtered_df[(price_filtered_df['Price_numeric'] == selected_price) &
                                     (price_filtered_df['Distance_numeric'] == selected_distance) &
                                     (price_filtered_df['Age'] == selected_age)]
    if selected_car.empty:
        return selected_car, price_lower_bound, price_upper_bound
    selected_car = selected_car.iloc[0]
    better_conditions = (
        (price_filtered_df['Mileage (ARAI)'].astype(float) > float(selected_car['Mileage (ARAI)'])) |
        (price_filtered_df['NCAP Rating'].astype(float) > float(selected_car['NCAP Rating'])) |
        (price_filtered_df['Airbags'].astype(float) > float(selected_car['Airbags'])) |
        (price_filtered_df['Seating Capacity'].astype(float) > float(selected_car['Seating Capacity'])) |
        (price_filtered_df['Seat Belt Warning'].astype(float) > float(selected_car['Seat Belt Warning'])) |
        (price_filtered_df['Transmission'].astype(float) > float(selected_car['Transmission'])) |
        (price_filtered_df['Distance_numeric'] < selected_distance) |
        (price_filtered_df['Age'] < selected_age)
    )
    better_options_df = price_filtered_df[better_conditions]
    better_options_df = better_options_df.sort_values(by=['Distance_numeric', 'Age'])
    return better_options_df, price_lower_bound, price_upper_bound

def new_suggestion_variants_selected_model(listed_df, make, model, selected_price, selected_distance, selected_age, price_range_percent=10):
    same_model_df = listed_df[(listed_df['Make'] == make) &
                              (listed_df['Model'] == model)].copy()
    if same_model_df.empty:
        return same_model_df, 0, 0
    price_lower_bound = selected_price * (1 - price_range_percent / 100)
    price_upper_bound = selected_price * (1 + price_range_percent / 100)
    price_filtered_df = same_model_df[(same_model_df['Price_numeric'] >= price_lower_bound) &
                                      (same_model_df['Price_numeric'] <= price_upper_bound)]
    if price_filtered_df.empty:
        return price_filtered_df, price_lower_bound, price_upper_bound
    selected_car = price_filtered_df[(price_filtered_df['Price_numeric'] == selected_price) &
                                     (price_filtered_df['Distance_numeric'] == selected_distance) &
                                     (price_filtered_df['Age'] == selected_age)]
    if selected_car.empty:
        return selected_car, price_lower_bound, price_upper_bound
    selected_car = selected_car.iloc[0]
    better_conditions = (
        (price_filtered_df['Mileage (ARAI)'].astype(float) > float(selected_car['Mileage (ARAI)'])) |
        (price_filtered_df['NCAP Rating'].astype(float) > float(selected_car['NCAP Rating'])) |
        (price_filtered_df['Airbags'].astype(float) > float(selected_car['Airbags'])) |
        (price_filtered_df['Seating Capacity'].astype(float) > float(selected_car['Seating Capacity'])) |
        (price_filtered_df['Seat Belt Warning'].astype(float) > float(selected_car['Seat Belt Warning'])) |
        (price_filtered_df['Transmission'].astype(float) > float(selected_car['Transmission'])) |
        (price_filtered_df['Distance_numeric'] < selected_distance) |
        (price_filtered_df['Age'] < selected_age)
    )
    better_options_df = price_filtered_df[better_conditions]
    better_options_df = better_options_df.sort_values(by=['Distance_numeric', 'Age'])
    return better_options_df, price_lower_bound, price_upper_bound

def generate_insight_no_restrictions(selected_car, better_options_data):
    better_options_df, price_lower_bound, price_upper_bound = better_options_data
    if better_options_df.empty:
        return {'status': 'no_better_options', 'comparisons': [], 'best_option': None, 'selected_car': selected_car,
                'price_lower_bound': price_lower_bound, 'price_upper_bound': price_upper_bound}
    best_option = better_options_df.iloc[0]
    comparisons = []
    if best_option['Mileage (ARAI)'] > selected_car['Mileage (ARAI)']:
        comparisons.append(f"better mileage ({best_option['Mileage (ARAI)']} km/l vs {selected_car['Mileage (ARAI)']} km/l)")
    if best_option['NCAP Rating'] > selected_car['NCAP Rating']:
        comparisons.append(f"higher NCAP Rating ({best_option['NCAP Rating']} vs {selected_car['NCAP Rating']})")
    if best_option['Airbags'] > selected_car['Airbags']:
        comparisons.append(f"more airbags ({int(best_option['Airbags'])} vs {int(selected_car['Airbags'])})")
    if best_option['Seating Capacity'] > selected_car['Seating Capacity']:
        comparisons.append(f"higher seating capacity ({int(best_option['Seating Capacity'])} vs {int(selected_car['Seating Capacity'])})")
    if best_option['Seat Belt Warning'] > selected_car['Seat Belt Warning']:
        comparisons.append("a seat belt warning feature")
    if best_option['Transmission'] > selected_car['Transmission']:
        comparisons.append("an automatic transmission")
    if best_option['Distance_numeric'] < selected_car['Distance_numeric']:
        comparisons.append(f"lower mileage ({int(best_option['Distance_numeric'])} km vs {int(selected_car['Distance_numeric'])} km)")
    if best_option['Age'] < selected_car['Age']:
        comparisons.append(f"a newer age ({int(best_option['Age'])} years vs {int(selected_car['Age'])} years)")
    return {
        'status': 'success',
        'comparisons': comparisons,
        'best_option': best_option,
        'selected_car': selected_car,
        'price_lower_bound': price_lower_bound,
        'price_upper_bound': price_upper_bound
    }

def finalize_insight_no_restrictions(listed_df, insight_data, yes_no_cols):
    if insight_data['status'] == 'no_better_options':
        return "No better options found across all cars in the same price range."
    best_option = insight_data['best_option']
    selected_car = insight_data['selected_car']
    comparisons = insight_data['comparisons']
    price_lower_bound = insight_data['price_lower_bound']
    price_upper_bound = insight_data['price_upper_bound']
    selected_car_row = listed_df[(listed_df['Make'] == selected_car['Make']) &
                                 (listed_df['Model'] == selected_car['Model']) &
                                 (listed_df['Variant'] == selected_car['Variant']) &
                                 (listed_df['Price_numeric'] == selected_car['Price_numeric']) &
                                 (listed_df['Distance_numeric'] == selected_car['Distance_numeric']) &
                                 (listed_df['Age'] == selected_car['Age'])]
    best_option_row = listed_df[(listed_df['Make'] == best_option['Make']) &
                                (listed_df['Model'] == best_option['Model']) &
                                (listed_df['Variant'] == best_option['Variant']) &
                                (listed_df['Price_numeric'] == best_option['Price_numeric']) &
                                (listed_df['Distance_numeric'] == best_option['Distance_numeric']) &
                                (listed_df['Age'] == best_option['Age'])]
    selected_car_feature_score = selected_car_row[yes_no_cols].sum(axis=1).iloc[0] if not selected_car_row.empty else 0
    best_option_feature_score = best_option_row[yes_no_cols].sum(axis=1).iloc[0] if not best_option_row.empty else 0
    if best_option_feature_score > selected_car_feature_score:
        comparisons.append(f"a higher feature score ({int(best_option_feature_score)} vs {int(selected_car_feature_score)})")
    if not comparisons:
        return "No significantly better options found across all cars based on the specified features."
    comparison_str = ", ".join(comparisons)
    insight = f"In the same price range, you can get a {best_option['Make']} {best_option['Model']} {best_option['Variant']} " \
              f"in {best_option['City']} with {comparison_str} for Rs {int(best_option['Price_numeric'])}. " \
              f"This is within the price range of Rs {int(price_lower_bound)} to Rs {int(price_upper_bound)}."
    return insight

def generate_insight_variants_selected_model(selected_car, better_options_data):
    better_options_df, price_lower_bound, price_upper_bound = better_options_data
    if better_options_df.empty:
        return {'status': 'no_better_options', 'comparisons': [], 'best_option': None, 'selected_car': selected_car,
                'price_lower_bound': price_lower_bound, 'price_upper_bound': price_upper_bound}
    best_option = better_options_df.iloc[0]
    comparisons = []
    if best_option['Mileage (ARAI)'] > selected_car['Mileage (ARAI)']:
        comparisons.append(f"better mileage ({best_option['Mileage (ARAI)']} km/l vs {selected_car['Mileage (ARAI)']} km/l)")
    if best_option['NCAP Rating'] > selected_car['NCAP Rating']:
        comparisons.append(f"higher NCAP Rating ({best_option['NCAP Rating']} vs {selected_car['NCAP Rating']})")
    if best_option['Airbags'] > selected_car['Airbags']:
        comparisons.append(f"more airbags ({int(best_option['Airbags'])} vs {int(selected_car['Airbags'])})")
    if best_option['Seating Capacity'] > selected_car['Seating Capacity']:
        comparisons.append(f"higher seating capacity ({int(best_option['Seating Capacity'])} vs {int(selected_car['Seating Capacity'])})")
    if best_option['Seat Belt Warning'] > selected_car['Seat Belt Warning']:
        comparisons.append("a seat belt warning feature")
    if best_option['Transmission'] > selected_car['Transmission']:
        comparisons.append("an automatic transmission")
    if best_option['Distance_numeric'] < selected_car['Distance_numeric']:
        comparisons.append(f"lower mileage ({int(best_option['Distance_numeric'])} km vs {int(selected_car['Distance_numeric'])} km)")
    if best_option['Age'] < selected_car['Age']:
        comparisons.append(f"a newer age ({int(best_option['Age'])} years vs {int(selected_car['Age'])} years)")
    return {
        'status': 'success',
        'comparisons': comparisons,
        'best_option': best_option,
        'selected_car': selected_car,
        'price_lower_bound': price_lower_bound,
        'price_upper_bound': price_upper_bound
    }

def finalize_insight_variants_selected_model(listed_df, insight_data, yes_no_cols):
    if insight_data['status'] == 'no_better_options':
        return "No better options found for different variants of this model in the same price range."
    best_option = insight_data['best_option']
    selected_car = insight_data['selected_car']
    comparisons = insight_data['comparisons']
    price_lower_bound = insight_data['price_lower_bound']
    price_upper_bound = insight_data['price_upper_bound']
    selected_car_row = listed_df[(listed_df['Make'] == selected_car['Make']) &
                                 (listed_df['Model'] == selected_car['Model']) &
                                 (listed_df['Variant'] == selected_car['Variant']) &
                                 (listed_df['Price_numeric'] == selected_car['Price_numeric']) &
                                 (listed_df['Distance_numeric'] == selected_car['Distance_numeric']) &
                                 (listed_df['Age'] == selected_car['Age'])]
    best_option_row = listed_df[(listed_df['Make'] == best_option['Make']) &
                                (listed_df['Model'] == best_option['Model']) &
                                (listed_df['Variant'] == best_option['Variant']) &
                                (listed_df['Price_numeric'] == best_option['Price_numeric']) &
                                (listed_df['Distance_numeric'] == best_option['Distance_numeric']) &
                                (listed_df['Age'] == best_option['Age'])]
    selected_car_feature_score = selected_car_row[yes_no_cols].sum(axis=1).iloc[0] if not selected_car_row.empty else 0
    best_option_feature_score = best_option_row[yes_no_cols].sum(axis=1).iloc[0] if not best_option_row.empty else 0
    if best_option_feature_score > selected_car_feature_score:
        comparisons.append(f"a higher feature score ({int(best_option_feature_score)} vs {int(selected_car_feature_score)})")
    if not comparisons:
        return "No significantly better options found for different variants of this model based on the specified features."
    comparison_str = ", ".join(comparisons)
    insight = f"In the same price range, you can get a {best_option['Make']} {best_option['Model']} {best_option['Variant']} " \
              f"in {best_option['City']} with {comparison_str} for Rs {int(best_option['Price_numeric'])}. " \
              f"This is within the price range of Rs {int(price_lower_bound)} to Rs {int(price_upper_bound)}."
    return insight