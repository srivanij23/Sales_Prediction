import pandas as pd

def save_predictions(df, predictions, output_path='outputs/predictions.csv'):
    df['predicted_car_purchase_amount'] = predictions
    df[['customer name', 'car purchase amount', 'predicted_car_purchase_amount']].to_csv(output_path, index=False)
