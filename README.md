# MPS Group Test - Titanic ML Pipeline

A complete machine learning pipeline using AWS SageMaker, CDK infrastructure, and the Titanic dataset for survival prediction. This project demonstrates a production-ready ML workflow with automated data processing, model training, and deployment.


## 📋 Prerequisites

- Python 3.8+
- AWS CLI configured with appropriate permissions
- Node.js and npm (for CDK)
- AWS CDK installed globally: `npm install -g aws-cdk`

## 🚀 Quick Start

### 1. Infrastructure Deployment

Deploy the AWS infrastructure using CDK:

```bash
# Navigate to CDK directory
cd cdk

# Install CDK dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy the stack
cdk deploy
```

This creates:
- **S3 Buckets**: Raw data, processed data, code storage, and model artifacts
- **IAM Roles**: SageMaker execution role with required permissions
- **Security**: Encrypted buckets with proper access controls

### 2. Install Python Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Execute ML Pipeline

Run the scripts in sequence:

```bash
# Step 1: Upload data and code to S3
python scripts/1.upload_all.py

# Step 2: Run complete ML pipeline (Processing → Training → Registration)
python scripts/2.create_processing_training_register.py

# Step 3: Deploy inference endpoint
python scripts/3.create_endpoint.py
```

## 📁 Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── cdk/                        # Infrastructure as Code
│   ├── app.py                  # CDK app entry point
│   ├── cdk_stack.py           # Stack definition
│   └── cdk.json               # CDK configuration
├── scripts/                    # Execution scripts
│   ├── 1.upload_all.py        # Data & code upload
│   ├── 2.create_processing_training_register.py  # ML pipeline
│   └── 3.create_endpoint.py   # Endpoint deployment
├── src/                        # Source code
│   ├── preprocesing/          # Data processing logic
│   │   └── preprocessing.py   # Feature engineering
│   └── training/              # Model training logic
│       └── train.py          # ML model training
└── data/                      # Dataset storage
    ├── titanic.csv           # Raw dataset
    └── processed/            # Processed data output
```

## 🔧 Script Functions

### 1. `1.upload_all.py` - Data & Code Upload

**Purpose**: Upload all necessary files to S3 buckets

**Functions**:
- Downloads Titanic dataset from public repository
- Uploads raw data to ingestion bucket with encryption
- Packages and uploads preprocessing code
- Creates training code archive (tar.gz format)
- Verifies all uploads completed successfully

**Output**: All required files available in S3 for processing

### 2. `2.create_processing_training_register.py` - ML Pipeline

**Purpose**: Execute complete ML pipeline with conditional execution

**Workflow**:
1. **Processing Job**: Clean and preprocess raw data
   - Feature engineering and data cleaning
   - Train/test split generation
   - Feature names extraction
2. **Training Job**: Train machine learning model (only if processing succeeds)
   - Random Forest classifier training
   - Hyperparameter optimization
   - Model artifact generation
3. **Model Registration**: Register trained model (only if training succeeds)
   - Create model package group
   - Register model version with metadata
   - Approve model for deployment

**Key Features**:
- Conditional execution: each step runs only if previous succeeds
- Comprehensive error handling and verification
- Optimized logging with essential status messages

### 3. `3.create_endpoint.py` - Endpoint Deployment

**Purpose**: Deploy trained model for real-time inference

**Functions**:
- Retrieves latest approved model from registry
- Creates SageMaker model configuration
- Deploys real-time inference endpoint
- Performs endpoint testing with sample data
- Provides inference examples and usage instructions

## 🔄 Workflow Details

### Step 1: Infrastructure Setup
```bash
cd cdk && cdk deploy
```
- Creates secure S3 buckets with encryption
- Sets up IAM roles with least-privilege access
- Establishes VPC and security groups (if needed)

### Step 2: Data & Code Preparation
```bash
python scripts/1.upload_all.py
```
- **Input**: Raw Titanic dataset (automatically downloaded)
- **Output**: Organized data and code in S3 buckets
- **Verification**: Confirms all uploads completed

### Step 3: Data Processing
```bash
python scripts/2.create_processing_training_register.py
```
**Processing Phase**:
- **Input**: Raw CSV data from S3
- **Processing**: Feature engineering, missing value handling, encoding
- **Output**: Clean train/test datasets + feature metadata
- **Duration**: ~5-10 minutes

### Step 4: Model Training  
**Training Phase** (runs automatically after processing):
- **Input**: Processed training data
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: Configurable via script
- **Output**: Trained model artifacts (model.tar.gz)
- **Duration**: ~10-15 minutes

### Step 5: Model Registration
**Registration Phase** (runs automatically after training):
- **Input**: Trained model artifacts
- **Process**: Version control, metadata tagging, approval
- **Output**: Approved model ready for deployment

### Step 6: Endpoint Deployment
```bash
python scripts/3.create_endpoint.py
```
- **Input**: Approved model from registry
- **Output**: Live inference endpoint
- **Testing**: Automatic endpoint validation
- **Duration**: ~5-8 minutes

## 🧪 Testing the Pipeline

After deployment, test the endpoint:

```python
import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Sample prediction data
test_data = "3,male,22.0,1,0,7.25,S,C"

# Make prediction
response = runtime.invoke_endpoint(
    EndpointName='titanic-endpoint-TIMESTAMP',
    ContentType='text/csv',
    Body=test_data
)

# Get prediction result
result = json.loads(response['Body'].read().decode())
print(f"Survival prediction: {result}")
```

## 🔧 Configuration

### Environment Variables
- `AWS_REGION`: AWS region for deployment (default: us-east-1)
- `STACK_NAME`: CDK stack name (default: CdkStack)

### Hyperparameters (Training)
Modify in `2.create_processing_training_register.py`:
```python
"HyperParameters": {
    "n-estimators": "100",    # Number of trees
    "max-depth": "10",        # Tree depth
    "min-samples-split": "5", # Min samples to split
    "min-samples-leaf": "2",  # Min samples per leaf
    "random-state": "42"      # Reproducibility
}
```

## 📊 Model Performance

The Random Forest model typically achieves:
- **Accuracy**: ~80-85% on test data
- **Features**: Age, Sex, Pclass, Fare, Embarked (encoded)
- **Target**: Binary survival prediction (0/1)

## 🧹 Cleanup

To avoid AWS charges, clean up resources:

```bash
# Delete SageMaker endpoints
python -c "
import boto3
sm = boto3.client('sagemaker')
endpoints = sm.list_endpoints()['Endpoints']
for ep in endpoints:
    if 'titanic' in ep['EndpointName']:
        sm.delete_endpoint(EndpointName=ep['EndpointName'])
        print(f'Deleted {ep[\"EndpointName\"]}')
"

# Destroy CDK stack
cd cdk && cdk destroy
```

