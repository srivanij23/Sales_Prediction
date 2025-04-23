import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_data
from src.modeling import train_models
from src.evaluation import evaluate_model
from src.utils import save_predictions

# Load data
df = pd.read_csv("data/car_purchasing.csv", encoding='latin1')

numerical_features = ['age', 'annual Salary', 'credit card debt', 'net worth']
categorical_features = ['country', 'gender']

X, y, preprocessor = preprocess_data(df.copy(), numerical_features, categorical_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = train_models(X_train, y_train)

# Evaluate
for name, model in models.items():
    predictions = model.predict(X_test)
    metrics = evaluate_model(y_test, predictions, name)

# Final predictions using best model
best_model = models['Gradient Boosting']
final_preds = best_model.predict(X)

# Save predictions
save_predictions(df.copy(), final_preds)
