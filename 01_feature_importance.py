# Feature Importance using Random Forest

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel("Feature_Importance.xlsx")

# Inputs and target
X = df[["Type of Vehicle", "Battery Capacity", "Driving Range"]]
y = df["Price"]

# Encode categorical
X_encoded = pd.get_dummies(X, columns=["Type of Vehicle"])

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

# Feature importance
importance_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf.feature_importances_
})

# Group features
importance_df["Base Feature"] = importance_df["Feature"].apply(
    lambda x: "Type of Vehicle" if "Type of Vehicle_" in x else x
)

grouped = importance_df.groupby("Base Feature", as_index=False)["Importance"].sum()
grouped = grouped.sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, y="Base Feature", x="Importance")
plt.title("Feature Importance")
plt.tight_layout()

# Save
plt.savefig("results/figures/feature_importance.png", dpi=300)
plt.show()