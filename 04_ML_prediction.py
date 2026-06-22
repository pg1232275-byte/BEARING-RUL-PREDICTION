import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# load features
feature_df = pd.read_csv(r'C:\Users\pg123\OneDrive\Desktop\nasa-bearing-rul\features.csv')
print(f"Dataset shape: {feature_df.shape}")
print(feature_df.columns.tolist())

# define features and target
FEATURES = ['rms', 'kurtosis', 'skewness', 'peak_to_peak', 'peak','standard deviation',
            'crest_factor', 'shape_factor', 'freq_energy']
X = feature_df[FEATURES].values
y = feature_df['RUL'].values

print(f"\nFeature matrix: {X.shape}")
print(f"RUL range: {y.min()} to {y.max()}")

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

# train and evaluate 3 models
models = {
    'Random Forest': RandomForestRegressor(
                     n_estimators=100, random_state=42),
    'SVR':           SVR(kernel='rbf', C=100, gamma=0.1),
    'KNN':           KNeighborsRegressor(n_neighbors=5)
}

results = {}
predictions_dict = {}

print("\n--- Model Results ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions_dict[name] = preds

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    cv  = cross_val_score(
              model, X_scaled, y, cv=5,
              scoring='r2').mean()

    results[name] = {
        'MAE (files)':  round(mae, 1),
        'MAE (hours)':  round(mae * 10 / 60, 2),
        'R2 Score':     round(r2, 4),
        'CV R2 (mean)': round(cv, 4)
    }
    print(f"{name}:")
    print(f"  MAE = {mae:.1f} files "
          f"({mae*10/60:.1f} hours)")
    print(f"  R2  = {r2:.4f}")
    print(f"  CV R2 = {cv:.4f}")

results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("\nModel comparison saved to model_results.csv")

# --- PLOT 3: Actual vs Predicted RUL ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, preds) in zip(
        axes, predictions_dict.items()):
    ax.scatter(y_test, preds,
               alpha=0.4, color='steelblue', s=15)
    ax.plot([y.min(), y.max()],
            [y.min(), y.max()],
            'r--', linewidth=2,
            label='Perfect prediction')
    ax.set_xlabel('Actual RUL (files)')
    ax.set_ylabel('Predicted RUL (files)')
    ax.set_title(f'{name}\nR2={results[name]["R2 Score"]}')
    ax.legend(fontsize=8)

plt.suptitle('RUL Prediction — Actual vs Predicted',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('plots/rul_prediction.png', dpi=150)
plt.close()
print("Plot 3 saved")

# --- PLOT 4: Feature importance ---
rf_model = models['Random Forest']
importances = rf_model.feature_importances_

imp_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(imp_df['Feature'], imp_df['Importance'],
         color='steelblue', edgecolor='white')
plt.xlabel('Importance Score')
plt.title('Feature Importance — What predicts '
          'bearing RUL best?')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("Plot 4 saved")

print("\n--- All done ---")
print("Check plots/ folder for all 4 plots")
print(results_df.to_string())