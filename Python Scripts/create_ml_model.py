import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = "/content/sample_data/exoplanet_database_new.csv"
df = pd.read_csv(file_path)

# Remove the outlier (if it's the highest value)
df = df[df['planet_temperature'] < df['planet_temperature'].max()]

df["log_orbit_semi_major_axis"] = np.log1p(df["orbit_semi_major_axis"])
# Define features (drop 'planet_temperature' which is the target)
X = df.drop(columns=['planet_temperature', 'planet_name', 'insolation_flux', 'eccentricity', 'Inclination']).values

# Define target variable
y = df['planet_temperature'].values

# Normalize features (important for deep learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model with the best hyperparameters from your search
model = Sequential([
    Dense(192, activation='relu', input_shape=(X_train.shape[1],)),  # units=192
    Dropout(0.2),  # dropout_rate=0.2
    Dense(128, activation='relu'),  # units_2=128
    Dropout(0.2),  # dropout_rate_2=0.2
    Dense(1)  # Output layer
])

# Compile the model with the best learning rate
model.compile(optimizer=Adam(learning_rate=0.0009), loss='mean_squared_error', metrics=['mae'])

# Display the model summary
model.summary()

# Train the model with the specified hyperparameters
history = model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))

# Predict on the test set
y_pred = model.predict(X_test)

# Compute evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Training Loss Curve")
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # Ideal line
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs. Predicted Planet Temperatures")
plt.show()

import joblib

# Save feature names before scaling
feature_columns = df.drop(columns=['planet_temperature', 'planet_name', 'insolation_flux' , 'eccentricity', 'Inclination']).columns
joblib.dump(feature_columns, "feature_columns_basic.pkl")

# Save the trained model
model.save("exoplanet_model_basic.h5")

# Save the scaler (used for feature normalization)
joblib.dump(scaler, "scalerbasic.pkl")

# Load new dataset
new_data = pd.read_csv("/content/sample_data/exoplanet_database_complete.csv", delimiter=';')

# Drop rows with missing values in required columns
new_data = new_data.dropna(subset=feature_columns)

# Extract only the required features (ensure correct order)
X_new = new_data[feature_columns]

# Transform using the pre-trained scaler
X_new_scaled = scaler.transform(X_new)

# Predict temperatures
y_pred = model.predict(X_new_scaled).flatten()

# Add predictions and keep original temperature for comparison
new_data["Predicted Temperature"] = y_pred

# Save results
new_data.to_csv("predicted_exoplanet_temperatures.csv", index=False)

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
