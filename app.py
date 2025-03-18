# app.py
import streamlit as st
from car_utils import (
    preprocess_data, find_yes_no_columns, convert_yes_no_to_binary,
    find_highest_features_same_variant, new_suggestion, generate_insight_same_variant,
    finalize_insight_with_feature_score, new_suggestion_different_variant,
    generate_insight_different_variant, finalize_insight_with_feature_score_different_variant,
    new_suggestion_no_restrictions, generate_insight_no_restrictions,
    finalize_insight_no_restrictions, new_suggestion_variants_selected_model,
    generate_insight_variants_selected_model, finalize_insight_variants_selected_model
)
import pandas as pd

st.title("Car Comparison App")
st.write("Select a car and compare it with better options based on your criteria.")

@st.cache_data
def load_data():
    file_path = "insights_final_merged_data.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Error: 'insights_final_merged_data.csv' not found. Please add it to the project folder.")
        return None
    return preprocess_data(df)

listed_df = load_data()

if listed_df is not None:
    st.subheader("Select Your Car Details")
    
    # Select Make
    make = st.selectbox("Make", options=sorted(listed_df['Make'].unique()))
    
    # Filter models based on selected Make
    model_options = sorted(listed_df[listed_df['Make'] == make]['Model'].unique())
    model = st.selectbox("Model", options=model_options)
    
    # Filter variants based on selected Make and Model
    variant_options = sorted(listed_df[(listed_df['Make'] == make) & 
                                       (listed_df['Model'] == model)]['Variant'].unique())
    variant = st.selectbox("Variant", options=variant_options)
    
    # Filter available prices based on Make, Model, and Variant
    filtered_df = listed_df[(listed_df['Make'] == make) & 
                            (listed_df['Model'] == model) & 
                            (listed_df['Variant'] == variant)]
    price_options = sorted(filtered_df['Price_numeric'].unique(), key=int)
    price = st.selectbox("Price (Rs)", options=price_options, format_func=lambda x: f"Rs {int(x):,}")
    
    # Filter available distances based on Make, Model, Variant, and Price
    filtered_df_by_price = filtered_df[filtered_df['Price_numeric'] == price]
    distance_options = sorted(filtered_df_by_price['Distance_numeric'].unique(), key=int)
    distance = st.selectbox("Distance Travelled (km)", options=distance_options, format_func=lambda x: f"{int(x):,} km")
    
    # Filter available ages based on Make, Model, Variant, Price, and Distance
    filtered_df_by_distance = filtered_df_by_price[filtered_df_by_price['Distance_numeric'] == distance]
    age_options = sorted(filtered_df_by_distance['Age'].unique(), key=int)
    age = st.selectbox("Age (years)", options=age_options, format_func=lambda x: f"{int(x)} years")
    
    # City selection remains unchanged
    city = st.selectbox("City", options=sorted(listed_df['City'].unique()))
    
    # Comparison type selection
    comparison_type = st.selectbox("Comparison Type", [
        "Same Variant",
        "Different Variants (Same Model)",
        "No Restrictions (All Cars)",
        "Different Variants of Selected Model"
    ])

    if st.button("Compare Cars"):
        selected_car = listed_df[(listed_df['Make'] == make) &
                                 (listed_df['Model'] == model) &
                                 (listed_df['Variant'] == variant) &
                                 (listed_df['Price_numeric'] == price) &
                                 (listed_df['Distance_numeric'] == distance) &
                                 (listed_df['Age'] == age)]
        
        if selected_car.empty:
            st.warning("No exact match found for this car. Adjust your inputs.")
        else:
            selected_car = selected_car.iloc[0]
            yes_no_cols = find_yes_no_columns(listed_df)
            st.subheader("Comparison Result")

            if comparison_type == "Same Variant":
                better_options = new_suggestion(listed_df, make, model, variant, city, price, distance, age)
                insight_data = generate_insight_same_variant(selected_car, better_options)
                final_insight = finalize_insight_with_feature_score(listed_df, insight_data, yes_no_cols)
                st.success(final_insight)

            elif comparison_type == "Different Variants (Same Model)":
                better_options = new_suggestion_different_variant(listed_df, make, model, city, price, distance, age)
                insight_data = generate_insight_different_variant(selected_car, better_options)
                final_insight = finalize_insight_with_feature_score_different_variant(listed_df, insight_data, yes_no_cols)
                st.success(final_insight)

            elif comparison_type == "No Restrictions (All Cars)":
                better_options = new_suggestion_no_restrictions(listed_df, price, distance, age)
                insight_data = generate_insight_no_restrictions(selected_car, better_options)
                final_insight = finalize_insight_no_restrictions(listed_df, insight_data, yes_no_cols)
                st.success(final_insight)

            elif comparison_type == "Different Variants of Selected Model":
                better_options = new_suggestion_variants_selected_model(listed_df, make, model, price, distance, age)
                insight_data = generate_insight_variants_selected_model(selected_car, better_options)
                final_insight = finalize_insight_variants_selected_model(listed_df, insight_data, yes_no_cols)
                st.success(final_insight)
else:
    st.stop()