import boto3
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

def verify_upload():
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

def main():
    """Upload preprocessing script to S3"""
    print("Uploading Code to S3")
    print("=" * 30)
    
    if upload_preprocessing_script():
        if verify_upload():
            print("Code upload completed successfully!")
        else:
            print("Upload completed but verification failed")
    else:
        print("Code upload failed")

if __name__ == "__main__":
    main()