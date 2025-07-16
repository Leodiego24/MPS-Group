import boto3
import pandas as pd
import requests
import tarfile
import tempfile
from pathlib import Path


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


# 1. upload_data
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


# 2. upload_code
def upload_preprocessing_script():
    """Upload preprocessing script to S3 code bucket"""
    print("Uploading preprocessing script to S3...")
    
    outputs = get_stack_outputs()
    if not outputs:
        return False
    
    code_bucket = outputs.get('CodeBucketName')
    if not code_bucket:
        print("Error: Code bucket name not found in outputs")
        return False
    
    # Check if preprocessing script exists
    script_path = Path("src/preprocesing/preprocessing.py")
    if not script_path.exists():
        print(f"Error: Preprocessing script not found at {script_path}")
        return False
    
    try:
        s3_client = boto3.client('s3')
        
        # Upload preprocessing script
        s3_client.upload_file(
            str(script_path),
            code_bucket,
            "preprocessing/preprocessing.py"
        )
        
        print(f"Script uploaded to: s3://{code_bucket}/preprocessing/preprocessing.py")
        return True
        
    except Exception as e:
        print(f"Error uploading script: {e}")
        return False


def verify_code_upload():
    """Verify script was uploaded successfully"""
    outputs = get_stack_outputs()
    if not outputs:
        return False
    
    code_bucket = outputs.get('CodeBucketName')
    if not code_bucket:
        return False
    
    try:
        s3_client = boto3.client('s3')
        
        # Check if script exists
        response = s3_client.head_object(
            Bucket=code_bucket,
            Key='preprocessing/preprocessing.py'
        )
        
        size_kb = response['ContentLength'] / 1024
        print(f"Upload verified: {size_kb:.1f} KB")
        return True
        
    except Exception as e:
        print(f"Upload verification failed: {e}")
        return False


# 3. upload_training_code
def create_training_package():
    """Create a tar.gz package with training code"""
    print("Creating training code package...")
    
    # Check if training script exists
    script_path = Path("src/training/train.py")
    if not script_path.exists():
        print(f"Error: Training script not found at {script_path}")
        return None
    
    # Create temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
        with tarfile.open(temp_file.name, 'w:gz') as tar:
            # Add the training script as train.py in the root of the archive
            tar.add(script_path, arcname='train.py')
            print(f"Added {script_path} to package as train.py")
        
        temp_path = Path(temp_file.name)
        print(f"Package created: {temp_path}")
        print(f"Package size: {temp_path.stat().st_size / 1024:.1f} KB")
        
        return temp_path


def upload_training_script():
    """Upload training script as tar.gz package to S3 code bucket"""
    print("Uploading training package to S3...")
    
    outputs = get_stack_outputs()
    if not outputs:
        return False
    
    code_bucket = outputs.get('CodeBucketName')
    if not code_bucket:
        print("Error: Code bucket name not found in outputs")
        return False
    
    # Create package
    package_path = create_training_package()
    if not package_path:
        return False
    
    try:
        s3_client = boto3.client('s3')
        
        # Upload the tar.gz package
        s3_key = "training/sourcedir.tar.gz"
        s3_client.upload_file(str(package_path), code_bucket, s3_key)
        
        print(f"Training package uploaded to: s3://{code_bucket}/{s3_key}")
        
        # Clean up temporary file
        package_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"Error uploading training package: {e}")
        return False


def verify_training_upload():
    """Verify package was uploaded successfully"""
    outputs = get_stack_outputs()
    if not outputs:
        return False
    
    code_bucket = outputs.get('CodeBucketName')
    if not code_bucket:
        return False
    
    try:
        s3_client = boto3.client('s3')
        
        # Check if package exists
        response = s3_client.head_object(
            Bucket=code_bucket,
            Key='training/sourcedir.tar.gz'
        )
        
        size_kb = response['ContentLength'] / 1024
        print(f"Upload verified: {size_kb:.1f} KB")
        return True
        
    except Exception as e:
        print(f"Upload verification failed: {e}")
        return False


def main():
    """Upload all files to S3"""
    print("Uploading All Files to S3")
    print("=" * 30)
    
    # 1. Upload data
    print("\n1. upload_data")
    print("-" * 20)
    print("Starting dataset configuration...")
    
    if upload_dataset_to_s3():
        print("Dataset uploaded successfully!")
    else:
        print("Error in dataset configuration")
    
    # 2. Upload code
    print("\n2. upload_code")
    print("-" * 20)
    print("Uploading Code to S3")
    
    if upload_preprocessing_script():
        if verify_code_upload():
            print("Code upload completed successfully!")
        else:
            print("Upload completed but verification failed")
    else:
        print("Code upload failed")
    
    # 3. Upload training code
    print("\n3. upload_training_code")
    print("-" * 25)
    print("Uploading Training Code to S3")
    
    if upload_training_script():
        if verify_training_upload():
            print("Training code upload completed successfully!")
            print("\nNext step: Create and run Training Job")
        else:
            print("Upload completed but verification failed")
    else:
        print("Training code upload failed")
    
    print("\n" + "=" * 30)
    print("All uploads completed!")


if __name__ == "__main__":
    main()