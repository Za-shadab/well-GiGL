import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class GlycemicIndexPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for the GI prediction model.
        Expected columns: carbohydrates, protein, fat, fiber, sugar
        """
        # Calculate derived features
        data['carb_to_fiber_ratio'] = data['carbohydrates'] / (data['fiber'] + 1)  # Add 1 to avoid division by zero
        data['sugar_percentage'] = data['sugar'] / data['carbohydrates'] * 100
        
        # Select features for the model
        features = [
            'carbohydrates',
            'protein',
            'fat',
            'fiber',
            'sugar',
            'carb_to_fiber_ratio',
            'sugar_percentage'
        ]
        
        return data[features]
    
    def train(self, X, y):
        """
        Train the GI prediction model
        
        Parameters:
        X: DataFrame with nutritional information
        y: Series with glycemic index values
        """
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        # Calculate and print model performance metrics
        scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"Cross-validation scores: {scores}")
        print(f"Average CV score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
    def predict(self, X):
        """
        Predict glycemic index for new food items
        
        Parameters:
        X: DataFrame with nutritional information
        
        Returns:
        Array of predicted GI values
        """
        X_processed = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        saved_objects = joblib.load(filepath)
        predictor = cls()
        predictor.model = saved_objects['model']
        predictor.scaler = saved_objects['scaler']
        return predictor

# Example usage
def main():
    # Sample data (you would need to replace this with real data)
    data = pd.DataFrame({
        'carbohydrates': [50, 25, 30, 45],
        'protein': [5, 20, 10, 8],
        'fat': [2, 15, 5, 3],
        'fiber': [3, 8, 4, 2],
        'sugar': [20, 5, 15, 25],
        'glycemic_index': [70, 45, 55, 65]
    })
    
    # Split features and target
    X = data.drop('glycemic_index', axis=1)
    y = data['glycemic_index']
    
    # Create and train the model
    gi_predictor = GlycemicIndexPredictor()
    gi_predictor.train(X, y)
    
    # Make predictions for new foods
    new_food = pd.DataFrame({
        'carbohydrates': [35],
        'protein': [12],
        'fat': [4],
        'fiber': [5],
        'sugar': [18]
    })
    
    predicted_gi = gi_predictor.predict(new_food)
    print(f"\nPredicted Glycemic Index: {predicted_gi[0]:.1f}")
    
    # Save the model
    gi_predictor.save_model('gi_predictor_model.joblib')

if __name__ == "__main__":
    main()