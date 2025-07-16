import boto3
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

def verify_upload():
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
    """Upload training script to S3"""
    print("Uploading Training Code to S3")
    print("=" * 35)
    
    if upload_training_script():
        if verify_upload():
            print("Training code upload completed successfully!")
            print("\nNext step: Create and run Training Job")
        else:
            print("Upload completed but verification failed")
    else:
        print("Training code upload failed")

if __name__ == "__main__":
    main()