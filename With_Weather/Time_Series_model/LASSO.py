from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create LASSO model with cross-validation
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Print coefficients
print("Feature Coefficients:")
for i, feature in enumerate(predictors):
    print(f"{feature}: {lasso.coef_[i]}")

# Get selected features
selected_lasso_features = [predictors[i] for i in range(len(predictors)) if lasso.coef_[i] != 0]
print("Selected LASSO Features:", selected_lasso_features)
