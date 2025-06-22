import joblib

# Adjust this path if needed
model_path = r"C:\Users\IfeomaAugustaAdigwe\Desktop\creditscore\src\artifacts\train_test_split.pkl"

# Load the model
model = joblib.load(model_path)

# Display the expected feature names if available
if hasattr(model, "feature_names_in_"):
	print(model.feature_names_in_)
else:
	print("The loaded model does not have the attribute 'feature_names_in_'.")

