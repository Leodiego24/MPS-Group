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


# 1. Processing Job Functions
def verify_processing_prerequisites():
    """Verify all prerequisites are met before creating processing job"""
    outputs = get_stack_outputs()
    if not outputs:
        print("✗ Could not get stack outputs")
        return False, None
    
    # Get required values
    ingestion_bucket = outputs.get('IngestionBucketName')
    processed_bucket = outputs.get('ProcessedDataBucketName')
    code_bucket = outputs.get('CodeBucketName')
    sagemaker_role = outputs.get('SageMakerRoleArn')
    
    if not all([ingestion_bucket, processed_bucket, code_bucket, sagemaker_role]):
        print("✗ Missing required stack outputs")
        return False, None
    
    # Verify S3 resources exist
    s3_client = boto3.client('s3')
    
    # Check ingestion bucket has data
    try:
        s3_client.head_object(Bucket=ingestion_bucket, Key='titanic/titanic.csv')
    except Exception:
        print("✗ Input data not found. Run: python scripts/1.upload_all.py")
        return False, None
    
    # Check code bucket has preprocessing script
    try:
        s3_client.head_object(Bucket=code_bucket, Key='preprocessing/preprocessing.py')
    except Exception:
        print("✗ Preprocessing script not found. Run: python scripts/1.upload_all.py")
        return False, None
    
    # Verify IAM role exists
    try:
        iam_client = boto3.client('iam')
        role_name = sagemaker_role.split('/')[-1]
        iam_client.get_role(RoleName=role_name)
    except Exception:
        print("✗ SageMaker role issue")
        return False, None
    
    print("✓ Processing prerequisites verified")
    return True, outputs


def create_processing_job(outputs):
    """Create SageMaker Processing Job"""
    # Get bucket names from outputs
    ingestion_bucket = outputs['IngestionBucketName']
    processed_bucket = outputs['ProcessedDataBucketName']
    code_bucket = outputs['CodeBucketName']
    sagemaker_role = outputs['SageMakerRoleArn']
    
    # Create unique job name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"titanic-preprocessing-{timestamp}"
    print(f"Creating processing job: {job_name}")
    
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
        response = sagemaker_client.create_processing_job(**processing_job_config)
        print(f"✓ Processing job created: {job_name}")
        return job_name
        
    except Exception as e:
        print(f"✗ Error creating processing job: {e}")
        return None


def monitor_processing_job(job_name):
    """Monitor processing job execution"""
    sagemaker_client = boto3.client('sagemaker')
    
    while True:
        try:
            response = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
            status = response['ProcessingJobStatus']
            
            if status == 'Completed':
                print("✓ Processing job completed")
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"✗ Processing job failed: {failure_reason}")
                return False
            elif status in ['Stopping', 'Stopped']:
                print(f"✗ Processing job stopped: {status}")
                return False
            elif status == 'InProgress':
                time.sleep(30)
            else:
                time.sleep(10)
                
        except Exception as e:
            print(f"✗ Error monitoring job: {e}")
            return False


def verify_processing_results(outputs):
    """Verify that processing job produced expected results"""
    processed_bucket = outputs['ProcessedDataBucketName']
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_objects_v2(Bucket=processed_bucket, Prefix='titanic/')
        
        if 'Contents' not in response:
            print("✗ No processed data found")
            return False
        
        # Check for expected files
        expected_files = ['train.csv', 'test.csv', 'feature_names.txt']
        found_files = [obj['Key'].split('/')[-1] for obj in response['Contents']]
        
        missing_files = [f for f in expected_files if f not in found_files]
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        
        print("✓ Processing results verified")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying results: {e}")
        return False


# 2. Training Job Functions
def verify_training_prerequisites():
    """Verify all prerequisites for training job"""
    outputs = get_stack_outputs()
    if not outputs:
        print("✗ Could not get stack outputs")
        return False, None
    
    # Get required values
    processed_bucket = outputs.get('ProcessedDataBucketName')
    model_bucket = outputs.get('ModelArtifactsBucketName')
    code_bucket = outputs.get('CodeBucketName')
    sagemaker_role = outputs.get('SageMakerRoleArn')
    
    if not all([processed_bucket, model_bucket, code_bucket, sagemaker_role]):
        print("✗ Missing required stack outputs")
        return False, None
    
    # Verify S3 resources exist
    s3_client = boto3.client('s3')
    
    # Check processed data exists
    try:
        s3_client.head_object(Bucket=processed_bucket, Key='titanic/train.csv')
        s3_client.head_object(Bucket=processed_bucket, Key='titanic/test.csv')
    except Exception:
        print("✗ Processed training data not found")
        return False, None
    
    # Check packaged training script exists
    try:
        s3_client.head_object(Bucket=code_bucket, Key='training/sourcedir.tar.gz')
    except Exception:
        print("✗ Training code not found. Run: python scripts/1.upload_all.py")
        return False, None
    
    print("✓ Training prerequisites verified")
    return True, outputs


def create_training_job(outputs):
    """Create SageMaker Training Job"""
    # Get bucket names from outputs
    processed_bucket = outputs['ProcessedDataBucketName']
    model_bucket = outputs['ModelArtifactsBucketName']
    code_bucket = outputs['CodeBucketName']
    sagemaker_role = outputs['SageMakerRoleArn']
    
    # Create unique job name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"titanic-training-{timestamp}"
    print(f"Creating training job: {job_name}")
    
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
        response = sagemaker_client.create_training_job(**training_job_config)
        print(f"✓ Training job created: {job_name}")
        return job_name
        
    except Exception as e:
        print(f"✗ Error creating training job: {e}")
        return None


def monitor_training_job(job_name):
    """Monitor training job execution"""
    sagemaker_client = boto3.client('sagemaker')
    
    while True:
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status == 'Completed':
                print("✓ Training job completed")
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"✗ Training job failed: {failure_reason}")
                return False
            elif status in ['Stopping', 'Stopped']:
                print(f"✗ Training job stopped: {status}")
                return False
            elif status == 'InProgress':
                time.sleep(30)
            else:
                time.sleep(10)
                
        except Exception as e:
            print(f"✗ Error monitoring training job: {e}")
            return False


def verify_training_results(outputs, job_name):
    """Verify that training job produced expected results"""
    model_bucket = outputs['ModelArtifactsBucketName']
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_objects_v2(Bucket=model_bucket, Prefix='titanic-model/')
        
        if 'Contents' not in response:
            print("✗ No model artifacts found")
            return False
        
        # Look for model.tar.gz
        model_files = [obj['Key'] for obj in response['Contents'] if 'model.tar.gz' in obj['Key']]
        if model_files:
            print("✓ Training results verified")
            return True
        else:
            print("✗ Model archive not found")
            return False
        
    except Exception as e:
        print(f"✗ Error verifying training results: {e}")
        return False


# 3. Model Registration Functions
def get_latest_training_job():
    """Get the latest completed training job"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            StatusEquals='Completed',
            MaxResults=10
        )
        
        # Look for titanic training jobs
        titanic_jobs = [job for job in response['TrainingJobSummaries'] 
                       if 'titanic' in job['TrainingJobName'].lower()]
        
        if not titanic_jobs:
            print("✗ No completed training jobs found")
            return None
        
        job_name = titanic_jobs[0]['TrainingJobName']
        print(f"Found training job: {job_name}")
        return job_name
        
    except Exception as e:
        print(f"✗ Error finding training job: {e}")
        return None


def get_training_job_details(job_name):
    """Get detailed information about the training job"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        return {
            'model_artifacts': response['ModelArtifacts']['S3ModelArtifacts'],
            'training_image': response['AlgorithmSpecification']['TrainingImage'],
            'role_arn': response['RoleArn'],
            'job_name': job_name
        }
        
    except Exception as e:
        print(f"✗ Error getting training job details: {e}")
        return None


def create_model_package_group():
    """Create a model package group for Titanic models"""
    sagemaker_client = boto3.client('sagemaker')
    group_name = "titanic-survival-model-group"
    
    try:
        # Check if group already exists
        try:
            sagemaker_client.describe_model_package_group(ModelPackageGroupName=group_name)
            return group_name
        except sagemaker_client.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise e
        
        # Create new group
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Model package group for Titanic survival prediction models"
        )
        
        print(f"✓ Created model package group: {group_name}")
        return group_name
        
    except Exception as e:
        print(f"✗ Error creating model package group: {e}")
        return None


def register_model_version(training_details, group_name):
    """Register a new model version"""
    sagemaker_client = boto3.client('sagemaker')
    
    # Create model package specification
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    model_package_spec = {
        "ModelPackageGroupName": group_name,
        "ModelPackageDescription": f"Titanic survival prediction model trained on {timestamp}",
        "ModelApprovalStatus": "PendingManualApproval",  # Will approve manually
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": training_details['training_image'],
                    "ModelDataUrl": training_details['model_artifacts'],
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "train.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                    }
                }
            ],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [
                "ml.t2.medium",
                "ml.m5.large",
                "ml.m5.xlarge"
            ],
            "SupportedTransformInstanceTypes": [
                "ml.m5.large",
                "ml.m5.xlarge"
            ]
        },
        "ModelMetrics": {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": training_details['model_artifacts'].replace('model.tar.gz', 'evaluation_results.json')
                }
            }
        },
        "CustomerMetadataProperties": {
            "TrainingJobName": training_details['job_name'],
            "ModelType": "RandomForestClassifier",
            "Dataset": "Titanic",
            "Framework": "scikit-learn",
            "TrainingTimestamp": timestamp
        }
    }
    
    try:
        response = sagemaker_client.create_model_package(**model_package_spec)
        model_package_arn = response['ModelPackageArn']
        print("✓ Model registered successfully")
        return model_package_arn
        
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return None


def approve_model(model_package_arn):
    """Approve the model for deployment"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription="Model approved for deployment after successful training"
        )
        
        print("✓ Model approved for deployment")
        return True
        
    except Exception as e:
        print(f"✗ Error approving model: {e}")
        return False


def main():
    """Main function to execute processing, training, and registration pipeline"""
    print("ML Pipeline: Processing → Training → Registration")
    
    # Step 1: Processing Job
    print("\n1. Processing Job")
    success, outputs = verify_processing_prerequisites()
    if not success:
        return
    
    processing_job_name = create_processing_job(outputs)
    if not processing_job_name:
        return
    
    processing_success = monitor_processing_job(processing_job_name)
    if not processing_success:
        print("✗ Processing failed. Cannot proceed.")
        return
    
    if not verify_processing_results(outputs):
        print("✗ Processing verification failed. Cannot proceed.")
        return
    
    # Step 2: Training Job (only if processing succeeded)
    print("\n2. Training Job")
    training_success, outputs = verify_training_prerequisites()
    if not training_success:
        return
    
    training_job_name = create_training_job(outputs)
    if not training_job_name:
        return
    
    training_success = monitor_training_job(training_job_name)
    if not training_success:
        print("✗ Training failed. Cannot proceed.")
        return
    
    if not verify_training_results(outputs, training_job_name):
        print("✗ Training verification failed. Cannot proceed.")
        return
    
    # Step 3: Model Registration (only if training succeeded)
    print("\n3. Model Registration")
    
    training_details = get_training_job_details(training_job_name)
    if not training_details:
        print("✗ Could not get training details")
        return
    
    group_name = create_model_package_group()
    if not group_name:
        return
    
    model_package_arn = register_model_version(training_details, group_name)
    if not model_package_arn:
        return
    
    approve_model(model_package_arn)
    
    # Final Success Message
    print(f"\n✓ ML Pipeline Completed Successfully!")
    print(f"Processing: {processing_job_name}")
    print(f"Training: {training_job_name}")
    print(f"Model: {group_name}")
    print("\nNext: Deploy endpoint with create_endpoint.py")


if __name__ == "__main__":
    main()