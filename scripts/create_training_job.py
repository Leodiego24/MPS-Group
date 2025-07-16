import boto3
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
    """Verify all prerequisites for training job"""
    print("Verifying prerequisites...")
    
    outputs = get_stack_outputs()
    if not outputs:
        print("Could not get stack outputs")
        return False, None
    
    # Get required values
    processed_bucket = outputs.get('ProcessedDataBucketName')
    model_bucket = outputs.get('ModelArtifactsBucketName')
    code_bucket = outputs.get('CodeBucketName')
    sagemaker_role = outputs.get('SageMakerRoleArn')
    
    if not all([processed_bucket, model_bucket, code_bucket, sagemaker_role]):
        print("Missing required outputs from stack")
        return False, None
    
    # Verify S3 resources exist
    s3_client = boto3.client('s3')
    
    # Check processed data exists
    try:
        s3_client.head_object(Bucket=processed_bucket, Key='titanic/train.csv')
        s3_client.head_object(Bucket=processed_bucket, Key='titanic/test.csv')
        print(f"✓ Processed training data found: s3://{processed_bucket}/titanic/")
    except Exception as e:
        print(f"✗ Processed training data not found: {e}")
        return False, None
    
    # Check packaged training script exists
    try:
        response = s3_client.head_object(Bucket=code_bucket, Key='training/sourcedir.tar.gz')
        print(f"✓ Packaged training code found: s3://{code_bucket}/training/sourcedir.tar.gz")
        print(f"   Size: {response['ContentLength']/1024:.1f} KB")
    except Exception as e:
        print(f"✗ Packaged training code not found: {e}")
        print("   Run: python scripts/package_training_code.py")
        return False, None
    
    print("✓ All prerequisites verified")
    return True, outputs

def create_training_job(outputs):
    """Create SageMaker Training Job"""
    print("\nCreating SageMaker Training Job...")
    
    # Get bucket names from outputs
    processed_bucket = outputs['ProcessedDataBucketName']
    model_bucket = outputs['ModelArtifactsBucketName']
    code_bucket = outputs['CodeBucketName']
    sagemaker_role = outputs['SageMakerRoleArn']
    
    # Create unique job name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"titanic-training-{timestamp}"
    
    # Training job configuration
    training_job_config = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            # Use the scikit-learn container
            "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "TrainingInputMode": "File"
        },
        "RoleArn": sagemaker_role,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{processed_bucket}/titanic/",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{model_bucket}/titanic-model/"
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600  # 1 hour max
        },
        "HyperParameters": {
            "sagemaker_program": "train.py",
            # FIXED: Point to the tar.gz file instead of directory
            "sagemaker_submit_directory": f"s3://{code_bucket}/training/sourcedir.tar.gz",
            "n-estimators": "100",
            "max-depth": "10",
            "min-samples-split": "5",
            "min-samples-leaf": "2",
            "random-state": "42"
        }
    }
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        
        print(f"Job name: {job_name}")
        print(f"Training data: s3://{processed_bucket}/titanic/")
        print(f"Code: s3://{code_bucket}/training/sourcedir.tar.gz")
        print(f"Model output: s3://{model_bucket}/titanic-model/")
        print(f"Instance: ml.m5.large")
        
        response = sagemaker_client.create_training_job(**training_job_config)
        
        print(f"\n✓ Training job created successfully!")
        print(f"Job ARN: {response['TrainingJobArn']}")
        
        return job_name
        
    except Exception as e:
        print(f"✗ Error creating training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor training job execution"""
    print(f"\nMonitoring training job: {job_name}")
    print("=" * 60)
    
    sagemaker_client = boto3.client('sagemaker')
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            
            status = response['TrainingJobStatus']
            elapsed_time = int(time.time() - start_time)
            
            # Get secondary status for more detail
            secondary_status = response.get('SecondaryStatus', 'Unknown')
            
            print(f"[{elapsed_time:3d}s] Status: {status} | {secondary_status}")
            
            if status == 'Completed':
                print(f"\n✓ Training job completed successfully!")
                
                # Show training time
                creation_time = response.get('CreationTime')
                end_time = response.get('TrainingEndTime')
                if creation_time and end_time:
                    duration = end_time - creation_time
                    print(f"Total training time: {duration}")
                
                # Show model location
                if 'ModelArtifacts' in response:
                    model_location = response['ModelArtifacts']['S3ModelArtifacts']
                    print(f"Model artifacts: {model_location}")
                
                return True
                
            elif status == 'Failed':
                print(f"\n✗ Training job failed!")
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"Failure reason: {failure_reason}")
                
                # Show CloudWatch logs URL
                region = boto3.Session().region_name or 'us-east-1'
                log_group = f'/aws/sagemaker/TrainingJobs'
                log_stream = job_name
                logs_url = f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{log_group}/log-events/{log_stream}"
                print(f"Check logs: {logs_url}")
                
                return False
                
            elif status in ['Stopping', 'Stopped']:
                print(f"\n⚠ Training job was stopped: {status}")
                return False
                
            elif status == 'InProgress':
                time.sleep(30)  # Wait 30 seconds for running jobs
            else:
                time.sleep(10)  # Wait 10 seconds for other statuses
                
        except Exception as e:
            print(f"Error monitoring job: {e}")
            return False

def verify_training_results(outputs, job_name):
    """Verify that training job produced expected results"""
    print("\nVerifying training results...")
    
    model_bucket = outputs['ModelArtifactsBucketName']
    s3_client = boto3.client('s3')
    
    try:
        # List objects in model bucket
        response = s3_client.list_objects_v2(
            Bucket=model_bucket,
            Prefix='titanic-model/'
        )
        
        if 'Contents' not in response:
            print("✗ No model artifacts found")
            return False
        
        print("✓ Model artifacts found:")
        total_size = 0
        for obj in response['Contents']:
            size_kb = obj['Size'] / 1024
            total_size += obj['Size']
            print(f"  {obj['Key']} ({size_kb:.1f} KB)")
        
        print(f"Total size: {total_size/1024:.1f} KB")
        
        # Look for model.tar.gz
        model_files = [obj['Key'] for obj in response['Contents'] if 'model.tar.gz' in obj['Key']]
        if model_files:
            print(f"✓ Model archive found: {model_files[0]}")
        else:
            print("⚠ Warning: model.tar.gz not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying results: {e}")
        return False

def main():
    """Main function to create and monitor training job"""
    print("SageMaker Training Job Creator (Fixed)")
    print("=" * 50)
    
    # Step 1: Verify prerequisites
    success, outputs = verify_prerequisites()
    if not success:
        print("\n✗ Prerequisites not met. Please fix issues above.")
        return
    
    # Step 2: Create training job
    job_name = create_training_job(outputs)
    if not job_name:
        print("\n✗ Failed to create training job")
        return
    
    # Step 3: Monitor job execution
    print(f"\nTraining job '{job_name}' is starting...")
    print("This may take 10-20 minutes depending on data size and model complexity.")
    
    success = monitor_training_job(job_name)
    
    if success:
        # Step 4: Verify results
        if verify_training_results(outputs, job_name):
            print("\n" + "=" * 50)
            print("✓ Training pipeline completed successfully!")
            print("=" * 50)
            print("Model trained and saved to S3")
            print("Ready for model registration and endpoint deployment")
            print("\nNext step: Register model and create endpoint")
        else:
            print("\n⚠ Training completed but results verification failed")
    else:
        print("\n✗ Training job failed")
        print("Check CloudWatch logs for detailed error information")

if __name__ == "__main__":
    main()