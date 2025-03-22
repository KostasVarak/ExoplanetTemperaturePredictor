import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = load_model("exoplanet_model_basic.h5")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Compile with correct optimizer and loss

# Load the trained scaler
scaler = joblib.load("scalerbasic.pkl")

# Load the feature columns used during training
feature_columns = joblib.load("feature_columns_basic.pkl")

# Load the new dataset (the one you want to make predictions on)
new_data = pd.read_csv("/content/solar_system.csv", delimiter=';')

# Check the columns in the dataset to ensure they match
print("Columns in the new dataset:")
print(new_data.columns)

# Drop rows with missing values in the required columns (those used in training)
new_data = new_data.dropna(subset=feature_columns)

# Extract the relevant features from the new dataset
X_new = new_data[feature_columns]

# Normalize using the pre-trained scaler (the scaler that was fitted on the training dataset)
X_new_scaled = scaler.transform(X_new)

# Predict the planet temperatures using the pre-trained model
y_pred = model.predict(X_new_scaled).flatten()

# Add the predictions to the original dataframe for comparison
new_data["Predicted Temperature"] = y_pred

# Save the results to a CSV file
new_data.to_csv("predicted_exoplanet_temperatures.csv", index=False)

# Display the first few rows of the dataframe with the predicted temperatures
print(new_data[['planet_temperature', 'Predicted Temperature']].head())

# --- Optional: If you have actual temperatures in the new data, you can also evaluate the predictions ---
if 'planet_temperature' in new_data.columns:
    # Identify valid rows where the actual temperature is available
    valid_indices = new_data['planet_temperature'].notna()

    if valid_indices.sum() > 0:
        # Get the actual temperatures and corresponding predicted temperatures
        y_actual = new_data.loc[valid_indices, 'planet_temperature'].values
        y_pred_filtered = new_data.loc[valid_indices, 'Predicted Temperature'].values

        # Compute evaluation metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_actual, y_pred_filtered)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred_filtered))
        r2 = r2_score(y_actual, y_pred_filtered)

        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² Score: {r2:.2f}")

    else:
        print("Warning: No valid actual planet temperature values for comparison.")
else:
    print("Warning: 'planet_temperature' column not found in the dataset.")
