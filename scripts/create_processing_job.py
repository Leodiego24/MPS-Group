import boto3
import json
import time
from datetime import datetime
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

def verify_prerequisites():
    """Verify all prerequisites are met before creating processing job"""
    print("Verifying prerequisites...")
    
    outputs = get_stack_outputs()
    if not outputs:
        print("Could not get stack outputs")
        return False, None
    
    # Get required values
    ingestion_bucket = outputs.get('IngestionBucketName')
    processed_bucket = outputs.get('ProcessedDataBucketName')
    code_bucket = outputs.get('CodeBucketName')
    sagemaker_role = outputs.get('SageMakerRoleArn')
    
    if not all([ingestion_bucket, processed_bucket, code_bucket, sagemaker_role]):
        print("Missing required outputs from stack")
        print(f"  Ingestion bucket: {ingestion_bucket}")
        print(f"  Processed bucket: {processed_bucket}")
        print(f"  Code bucket: {code_bucket}")
        print(f"  SageMaker role: {sagemaker_role}")
        return False, None
    
    # Verify S3 resources exist
    s3_client = boto3.client('s3')
    
    # Check ingestion bucket has data
    try:
        response = s3_client.head_object(Bucket=ingestion_bucket, Key='titanic/titanic.csv')
        print(f"Input data found: s3://{ingestion_bucket}/titanic/titanic.csv")
        print(f"   Size: {response['ContentLength']} bytes")
    except Exception as e:
        print(f"Input data not found: {e}")
        print("   Run: python scripts/upload_dataset.py")
        return False, None
    
    # Check code bucket has preprocessing script
    try:
        response = s3_client.head_object(Bucket=code_bucket, Key='preprocessing/preprocessing.py')
        print(f"Preprocessing script found: s3://{code_bucket}/preprocessing/preprocessing.py")
        print(f"   Size: {response['ContentLength']} bytes")
    except Exception as e:
        print(f"Preprocessing script not found: {e}")
        print("   Run: python scripts/upload_code.py")
        return False, None
    
    # Verify IAM role exists
    try:
        iam_client = boto3.client('iam')
        role_name = sagemaker_role.split('/')[-1]
        iam_client.get_role(RoleName=role_name)
        print(f"SageMaker role verified: {role_name}")
    except Exception as e:
        print(f"SageMaker role issue: {e}")
        return False, None
    
    print("All prerequisites verified")
    return True, outputs

def create_processing_job(outputs):
    """Create SageMaker Processing Job"""
    print("\nCreating SageMaker Processing Job...")
    
    # Get bucket names from outputs
    ingestion_bucket = outputs['IngestionBucketName']
    processed_bucket = outputs['ProcessedDataBucketName']
    code_bucket = outputs['CodeBucketName']
    sagemaker_role = outputs['SageMakerRoleArn']
    
    # Create unique job name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"titanic-preprocessing-{timestamp}"
    
    # Processing job configuration
    processing_job_config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.medium",
                "VolumeSizeInGB": 10
            }
        },
        "AppSpecification": {
            "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "ContainerEntrypoint": [
                "python3", 
                "/opt/ml/processing/input/code/preprocessing.py"
            ]
        },
        "RoleArn": sagemaker_role,
        "ProcessingInputs": [
            {
                "InputName": "input-data",
                "S3Input": {
                    "S3Uri": f"s3://{ingestion_bucket}/titanic/",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            },
            {
                "InputName": "code", 
                "S3Input": {
                    "S3Uri": f"s3://{code_bucket}/preprocessing/",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
            {
                "OutputName": "processed-data",
                "S3Output": {
                    "S3Uri": f"s3://{processed_bucket}/titanic/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob"
                }
            }
            ]
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600  # 1 hour max
        }
    }
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        
        print(f"Job name: {job_name}")
        print(f"Input data: s3://{ingestion_bucket}/titanic/")
        print(f"Code: s3://{code_bucket}/preprocessing/")
        print(f"Output: s3://{processed_bucket}/titanic/")
        print(f"Instance: ml.t3.medium")
        
        response = sagemaker_client.create_processing_job(**processing_job_config)
        
        print(f"\nProcessing job created successfully!")
        print(f"Job ARN: {response['ProcessingJobArn']}")
        
        return job_name
        
    except Exception as e:
        print(f"Error creating processing job: {e}")
        return None

def monitor_processing_job(job_name):
    """Monitor processing job execution"""
    print(f"\nMonitoring processing job: {job_name}")
    print("=" * 60)
    
    sagemaker_client = boto3.client('sagemaker')
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker_client.describe_processing_job(
                ProcessingJobName=job_name
            )
            
            status = response['ProcessingJobStatus']
            elapsed_time = int(time.time() - start_time)
            
            print(f"[{elapsed_time:3d}s] Status: {status}")
            
            if status == 'Completed':
                print(f"\nProcessing job completed successfully!")
                
                # Show processing time
                creation_time = response.get('CreationTime')
                end_time = response.get('ProcessingEndTime')
                if creation_time and end_time:
                    duration = end_time - creation_time
                    print(f"Total processing time: {duration}")
                
                # Show resource usage
                if 'ProcessingResources' in response:
                    resources = response['ProcessingResources']
                    print(f"Instance type: {resources['ClusterConfig']['InstanceType']}")
                    print(f"Instance count: {resources['ClusterConfig']['InstanceCount']}")
                
                return True
                
            elif status == 'Failed':
                print(f"\nProcessing job failed!")
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"Failure reason: {failure_reason}")
                
                # Show exit message if available
                if 'ExitMessage' in response:
                    print(f"Exit message: {response['ExitMessage']}")
                
                return False
                
            elif status in ['Stopping', 'Stopped']:
                print(f"\nProcessing job was stopped: {status}")
                return False
                
            elif status == 'InProgress':
                time.sleep(30)  # Wait 30 seconds for running jobs
            else:
                time.sleep(10)  # Wait 10 seconds for other statuses
                
        except Exception as e:
            print(f"Error monitoring job: {e}")
            return False

def verify_processing_results(outputs):
    """Verify that processing job produced expected results"""
    print("\nVerifying processing results...")
    
    processed_bucket = outputs['ProcessedDataBucketName']
    s3_client = boto3.client('s3')
    
    try:
        # List objects in processed data bucket
        response = s3_client.list_objects_v2(
            Bucket=processed_bucket,
            Prefix='titanic/'
        )
        
        if 'Contents' not in response:
            print("No processed data found")
            return False
        
        print("Processed files found:")
        total_size = 0
        for obj in response['Contents']:
            size_kb = obj['Size'] / 1024
            total_size += obj['Size']
            print(f"  {obj['Key']} ({size_kb:.1f} KB)")
        
        print(f"Total size: {total_size/1024:.1f} KB")
        
        # Check for expected files
        expected_files = ['train.csv', 'test.csv', 'feature_names.txt']
        found_files = [obj['Key'].split('/')[-1] for obj in response['Contents']]
        
        missing_files = [f for f in expected_files if f not in found_files]
        if missing_files:
            print(f"Missing expected files: {missing_files}")
        else:
            print("All expected files found!")
        
        return True
        
    except Exception as e:
        print(f"Error verifying results: {e}")
        return False

def download_processing_results(outputs):
    """Download processed data for local inspection"""
    print("\nDownloading processed data...")
    
    processed_bucket = outputs['ProcessedDataBucketName']
    s3_client = boto3.client('s3')
    
    # Create local directory
    local_dir = Path("data/processed/sagemaker")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        ('titanic/train.csv', 'train.csv'),
        ('titanic/test.csv', 'test.csv'),
        ('titanic/feature_names.txt', 'feature_names.txt')
    ]
    
    try:
        for s3_key, local_filename in files_to_download:
            try:
                local_path = local_dir / local_filename
                s3_client.download_file(processed_bucket, s3_key, str(local_path))
                
                # Get file size
                size_kb = local_path.stat().st_size / 1024
                print(f"  {local_filename} ({size_kb:.1f} KB)")
                
            except Exception as e:
                print(f"  Could not download {local_filename}: {e}")
        
        print(f"\nFiles saved to: {local_dir}")
        
        # Show data summary if possible
        try:
            import pandas as pd
            
            train_df = pd.read_csv(local_dir / 'train.csv')
            test_df = pd.read_csv(local_dir / 'test.csv')
            
            print(f"\nData Summary:")
            print(f"  Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
            print(f"  Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
            
            if 'Survived' in train_df.columns:
                survival_rate = train_df['Survived'].mean()
                print(f"  Survival rate: {survival_rate:.3f}")
            
            # Show feature list
            feature_file = local_dir / 'feature_names.txt'
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    features = f.read().strip().split('\n')
                print(f"  Features ({len(features)}): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
                
        except ImportError:
            print("  Install pandas to see detailed data summary")
        except Exception as e:
            print(f"  Could not analyze data: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        return False

def main():
    """Main function to create and monitor processing job"""
    print("SageMaker Processing Job Creator")
    print("=" * 50)
    
    # Step 1: Verify prerequisites
    success, outputs = verify_prerequisites()
    if not success:
        print("\nPrerequisites not met. Please fix issues above.")
        return
    
    # Step 2: Create processing job
    job_name = create_processing_job(outputs)
    if not job_name:
        print("\nFailed to create processing job")
        return
    
    # Step 3: Monitor job execution
    print(f"\nProcessing job '{job_name}' is starting...")
    print("This may take 5-15 minutes depending on data size and complexity.")
    
    success = monitor_processing_job(job_name)
    
    if success:
        # Step 4: Verify and download results
        if verify_processing_results(outputs):
            download_processing_results(outputs)
            
            print("\n" + "=" * 50)
            print("Processing pipeline completed successfully!")
            print("=" * 50)
            print("Data preprocessed and saved to S3")
            print("Results downloaded locally for inspection")
            print("Check: data/processed/sagemaker/")
            print("\nNext step: Proceed to training phase")
        else:
            print("\nProcessing completed but results verification failed")
    else:
        print("\nProcessing job failed")
        print("Check CloudWatch logs for detailed error information")
        print("Verify input data and preprocessing script")

if __name__ == "__main__":
    main()