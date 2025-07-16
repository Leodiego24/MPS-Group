import boto3
import pandas as pd
import requests
from pathlib import Path
import json

def get_stack_outputs():
    """Get CDK stack outputs"""
    try:
        cf_client = boto3.client('cloudformation')
        response = cf_client.describe_stacks(StackName='CdkStack')
        
        outputs = {}
        for output in response['Stacks'][0]['Outputs']:
            outputs[output['OutputKey']] = output['OutputValue']
        
        return outputs
    except Exception as e:
        print(f"Error getting stack outputs: {e}")
        return None

def download_titanic_dataset():
    """Download Titanic dataset"""
    
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        dataset_path = data_dir / "titanic.csv"
        with open(dataset_path, "w") as f:
            f.write(response.text)
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def upload_dataset_to_s3():
    """Upload dataset to S3 ingestion bucket with restricted permissions"""
    
    # Get stack configuration
    outputs = get_stack_outputs()
    if not outputs:
        return False
    
    # Download dataset if it doesn't exist
    dataset_path = Path("data/titanic.csv")
    if not dataset_path.exists():
        dataset_path = download_titanic_dataset()
        if not dataset_path:
            return False
    
    # Configure S3 client
    s3_client = boto3.client('s3')
    bucket_name = outputs.get('IngestionBucketName')
    
    if not bucket_name:
        return False
    
    try:
        # Upload file with metadata
        s3_client.upload_file(
            str(dataset_path),
            bucket_name,
            "titanic/titanic.csv",
            ExtraArgs={
                'Metadata': {
                    'dataset-type': 'titanic',
                    'data-classification': 'raw',
                    'upload-date': str(pd.Timestamp.now())
                },
                'ServerSideEncryption': 'AES256'
            }
        )
        print(f"Location: s3://{bucket_name}/titanic/titanic.csv")
        
        return True
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return False

if __name__ == "__main__":
    print("Starting dataset configuration...")
    
    if upload_dataset_to_s3():
        print("Dataset uploaded successfully!")
    else:
        print("Error in dataset configuration")