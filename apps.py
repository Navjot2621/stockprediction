import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title of the app
st.title("Stock Market Price Prediction")

# File Upload section
st.sidebar.header("Upload your Stock Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Display data overview
    st.header("Data Overview")
    st.write(df.head(10))

    # Check for null values
    if df.isnull().sum().any():
        st.warning("Data contains missing values. Handling null values...")
        df = df.dropna()  # Or use df.fillna() if you want to impute the values

    # Feature Engineering
    st.header("Feature Engineering")
    st.write("Using 'Date' column as index if available and preparing 'Open', 'High', 'Low', and 'Close' prices for prediction")

    # Convert 'Date' to datetime, handle errors with 'coerce' and day-first format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    # Check if there are any invalid dates
    invalid_dates = df[df['Date'].isna()]
    if not invalid_dates.empty:
        st.warning(f"Some rows contain invalid dates:\n{invalid_dates[['Date']]}")

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Feature selection (predicting 'Close' price as an example)
    X = df[['Open', 'High', 'Low', 'Volume']]  # Use other features as required
    y = df['Close']  # Target: 'Close' price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection (user choice from sidebar)
    st.sidebar.header("Select Model")
    model_choice = st.sidebar.selectbox("Choose model", ("Linear Regression", "Decision Tree", "Random Forest"))

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
    }

    # Select the model
    selected_model = models[model_choice]

    # Training the model
    selected_model.fit(X_train, y_train)

    # Predictions
    y_pred = selected_model.predict(X_test)

    # Performance metrics
    st.subheader("Model Performance Metrics")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**R² Score**: {r2:.2f}")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")

    # Plotting
    st.subheader("Model Prediction vs Actual")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual", color="blue")
    ax.plot(y_pred, label="Predicted", color="red", linestyle='--')
    ax.legend()
    st.pyplot(fig)

    # User input for prediction
    st.write("Enter values for prediction:")
    user_input = {}
    for column in X.columns:
        user_input[column] = st.number_input(
            column,
            min_value=float(np.min(X[column])),
            max_value=float(np.max(X[column])),
            value=float(np.mean(X[column]))  # Default value is the column mean
        )

    user_input_df = pd.DataFrame([user_input])
    user_input_sc_df = scaler.transform(user_input_df)

    # Predicting the stock price for the entered values
    predicted_price = selected_model.predict(user_input_sc_df)

    st.subheader("Predicted Stock Price:")
    st.write(f"${predicted_price[0]:,.2f}")
