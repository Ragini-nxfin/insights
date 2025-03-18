# app.py
import streamlit as st
import warnings
warnings.filterwarnings('ignore')





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
    make = st.selectbox("Make", options=sorted(listed_df['Make'].unique()))
    model = st.selectbox("Model", options=sorted(listed_df[listed_df['Make'] == make]['Model'].unique()))
    variant = st.selectbox("Variant", options=sorted(listed_df[(listed_df['Make'] == make) & (listed_df['Model'] == model)]['Variant'].unique()))
    city = st.selectbox("City", options=sorted(listed_df['City'].unique()))
    price = st.number_input("Price (Rs)", min_value=0, value=500000, step=10000)
    distance = st.number_input("Distance Travelled (km)", min_value=0, value=50000, step=1000)
    age = st.number_input("Age (years)", min_value=0, value=3, step=1)

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