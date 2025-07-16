import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import boto3

def load_data(input_path):
    """Load training and test data"""
    print("Loading training and test data...")
    
    train_path = os.path.join(input_path, 'train.csv')
    test_path = os.path.join(input_path, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def prepare_features_target(df):
    """Separate features and target"""
    if 'Survived' not in df.columns:
        raise ValueError("Target column 'Survived' not found in data")
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_model(X_train, y_train, model_params):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        random_state=model_params['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"Model trained with {model.n_estimators} trees")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    return metrics, y_pred, y_pred_proba

def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    print("Calculating feature importance...")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def save_model_and_artifacts(model, metrics, feature_importance, feature_names, output_path):
    """Save model and related artifacts"""
    print(f"Saving model and artifacts to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(output_path, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save feature importance
    importance_path = os.path.join(output_path, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Save feature names
    features_path = os.path.join(output_path, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {features_path}")
    
    # Save model info
    model_info = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'n_features': len(feature_names),
        'training_accuracy': metrics['accuracy']
    }
    
    info_path = os.path.join(output_path, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {info_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Titanic survival prediction model')
    
    # SageMaker paths
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/output/data')
    
    # Model hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=5)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--random-state', type=int, default=42)
    
    args = parser.parse_args()
    
    print("Starting Titanic model training...")
    print(f"Training data path: {args.train}")
    print(f"Model output path: {args.model_dir}")
    print(f"Output data path: {args.output_data_dir}")
    
    try:
        # Load data
        train_df, test_df = load_data(args.train)
        
        # Prepare features and target
        X_train, y_train = prepare_features_target(train_df)
        X_test, y_test = prepare_features_target(test_df)
        
        # Ensure same features in train and test
        if list(X_train.columns) != list(X_test.columns):
            print("Warning: Feature mismatch between train and test")
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            print(f"Using {len(common_features)} common features")
        
        # Model parameters
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'random_state': args.random_state
        }
        
        print(f"Model parameters: {model_params}")
        
        # Train model
        model = train_model(X_train, y_train, model_params)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
        
        # Get feature importance
        feature_names = list(X_train.columns)
        feature_importance = get_feature_importance(model, feature_names)
        
        # Save model and artifacts
        save_model_and_artifacts(
            model, metrics, feature_importance, feature_names, args.model_dir
        )
        
        # Save evaluation results
        evaluation_results = {
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'test_predictions': y_pred.tolist(),
            'test_probabilities': y_pred_proba.tolist()
        }
        
        eval_path = os.path.join(args.output_data_dir, 'evaluation_results.json')
        os.makedirs(args.output_data_dir, exist_ok=True)
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved to: {eval_path}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()