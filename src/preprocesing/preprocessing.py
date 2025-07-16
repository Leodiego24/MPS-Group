import pandas as pd
import numpy as np
import argparse
import os
import boto3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Clean the Titanic dataset by handling missing values
    """
    print("Starting data cleaning...")
    
    # Handle missing values in Age - fill with median
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print(f"   Filled {df['Age'].isnull().sum()} missing Age values with median: {median_age:.1f}")
    
    # Handle missing values in Embarked - fill with mode
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)
    print(f"   Filled missing Embarked values with mode: {mode_embarked}")
    
    # Handle missing values in Fare - fill with median
    if df['Fare'].isnull().sum() > 0:
        median_fare = df['Fare'].median()
        df['Fare'].fillna(median_fare, inplace=True)
        print(f"   Filled missing Fare values with median: {median_fare:.2f}")
    
    # Drop Cabin column (too many missing values)
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)
        print("   Dropped Cabin column (too many missing values)")
    
    # Drop unnecessary columns
    columns_to_drop = ['Name', 'Ticket', 'PassengerId']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(existing_columns_to_drop, axis=1, inplace=True)
        print(f"   Dropped unnecessary columns: {existing_columns_to_drop}")
    
    print(f"Data after cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def feature_engineering(df):
    """
    Create new features and transform existing ones
    """
    print("Starting feature engineering...")
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print("   Created FamilySize feature")
    
    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print("   Created IsAlone feature")
    
    # Create Age groups
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 18, 35, 60, 100], 
                           labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    print("   Created AgeGroup feature")
    
    # Create Fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], 
                             q=4, 
                             labels=['Low', 'Medium', 'High', 'Very High'])
    print("   Created FareGroup feature")
    
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables for machine learning
    """
    print("Starting categorical encoding...")
    
    # Label encode binary categorical variables
    label_encoder = LabelEncoder()
    
    # Encode Sex
    df['Sex_encoded'] = label_encoder.fit_transform(df['Sex'])
    print("   Encoded Sex variable")
    
    # One-hot encode Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    print(f"   One-hot encoded Embarked: {list(embarked_dummies.columns)}")
    
    # One-hot encode Pclass
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, pclass_dummies], axis=1)
    print(f"   One-hot encoded Pclass: {list(pclass_dummies.columns)}")
    
    # One-hot encode AgeGroup
    agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
    df = pd.concat([df, agegroup_dummies], axis=1)
    print(f"   One-hot encoded AgeGroup: {list(agegroup_dummies.columns)}")
    
    # One-hot encode FareGroup
    faregroup_dummies = pd.get_dummies(df['FareGroup'], prefix='FareGroup')
    df = pd.concat([df, faregroup_dummies], axis=1)
    print(f"   One-hot encoded FareGroup: {list(faregroup_dummies.columns)}")
    
    # Drop original categorical columns
    categorical_columns = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup']
    df.drop(categorical_columns, axis=1, inplace=True)
    print(f"   Dropped original categorical columns: {categorical_columns}")
    
    return df

def normalize_numerical_features(df):
    """
    Normalize numerical features using StandardScaler
    """
    print("Starting numerical normalization...")
    
    # Identify numerical columns (excluding target and encoded categorical)
    numerical_columns = ['Age', 'Fare', 'FamilySize']
    existing_numerical = [col for col in numerical_columns if col in df.columns]
    
    if existing_numerical:
        scaler = StandardScaler()
        df[existing_numerical] = scaler.fit_transform(df[existing_numerical])
        print(f"   Normalized numerical features: {existing_numerical}")
        
        # Save scaler for later use in inference
        return df, scaler
    else:
        print("Warning: No numerical features found to normalize")
        return df, None

def save_processed_data(train_df, test_df, output_path):
    """
    Save processed data to S3 or local path
    """
    print(f"Saving processed data to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save train and test datasets
    train_path = os.path.join(output_path, 'train.csv')
    test_path = os.path.join(output_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   Saved train data: {train_path} ({train_df.shape[0]} rows)")
    print(f"   Saved test data: {test_path} ({test_df.shape[0]} rows)")
    
    # Save feature names for later use
    feature_names_path = os.path.join(output_path, 'feature_names.txt')
    feature_names = [col for col in train_df.columns if col != 'Survived']
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"   Saved feature names: {feature_names_path}")

def main():
    """
    Main preprocessing function
    """
    parser = argparse.ArgumentParser(description='Preprocess Titanic dataset')
    parser.add_argument('--input-path', type=str, default='/opt/ml/processing/input/data',
                       help='Path to input data')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/output',
                       help='Path to save processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("Starting Titanic data preprocessing...")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    
    # Load data
    input_file = os.path.join(args.input_path, 'titanic.csv')
    if not os.path.exists(input_file):
        # Try alternative path structure
        input_file = os.path.join(args.input_path, 'titanic', 'titanic.csv')
    
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Display basic info about missing values
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("Missing values per column:")
        for col, missing_count in missing_info[missing_info > 0].items():
            percentage = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count} ({percentage:.1f}%)")
    
    # Apply preprocessing steps
    df_clean = clean_data(df.copy())
    df_engineered = feature_engineering(df_clean)
    df_encoded = encode_categorical_variables(df_engineered)
    df_normalized, scaler = normalize_numerical_features(df_encoded)
    
    # Split into train and test sets
    print(f"Splitting data (test_size={args.test_size}, random_state={args.random_state})")
    
    # Separate features and target
    if 'Survived' in df_normalized.columns:
        X = df_normalized.drop('Survived', axis=1)
        y = df_normalized['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state,
            stratify=y
        )
        
        # Combine features and target back
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"Train set: {train_df.shape[0]} rows")
        print(f"Test set: {test_df.shape[0]} rows")
        print(f"Survival rate in train: {y_train.mean():.3f}")
        print(f"Survival rate in test: {y_test.mean():.3f}")
        
    else:
        print("Warning: 'Survived' column not found, saving full dataset as train")
        train_df = df_normalized
        test_df = pd.DataFrame()  # Empty test set
    
    # Save processed data
    save_processed_data(train_df, test_df, args.output_path)
    
    print("Preprocessing completed successfully!")
    print(f"Final processed data shape: {train_df.shape}")
    print(f"Final features: {[col for col in train_df.columns if col != 'Survived']}")

if __name__ == "__main__":
    main()