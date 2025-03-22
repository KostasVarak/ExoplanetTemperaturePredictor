import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import keras_tuner as kt  # Import KerasTuner

# Load the cleaned dataset
file_path = "/content/exoplanet_database_filtered.csv"
df = pd.read_csv(file_path)

# Remove the outlier (if it's the highest value)
df = df[df['planet_temperature'] < df['planet_temperature'].max()]

# Define features (drop 'planet_temperature' which is the target)
X = df.drop(columns=['planet_temperature']).values
X = df.drop(columns=['planet_name']).values

# Define target variable
y = df['planet_temperature'].values

# Normalize features (important for deep learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to create the model for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=64), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1))  # Output layer

    # Compile the model with a tunable learning rate
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.001, step=0.0001)),
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model

# Initialize the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_mae',  # Objective to minimize
    max_epochs=10,  # Limit to a number of epochs
    hyperband_iterations=2,  # Number of iterations to run
    directory='my_dir',  # Where to save the logs and search results
    project_name='exoplanet_tuning'  # Name of the project
)

# Start the search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Retrieve the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")

# Build the best model with the found hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
best_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Predict on the test set
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the training history
import matplotlib.pyplot as plt
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

# SHAP for feature importance
import shap

explainer = shap.KernelExplainer(best_model.predict, X_train)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=df.drop(columns=['planet_temperature', 'planet_name']).columns)
