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
    print("Verifying processing prerequisites...")
    
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
        print("   Run: python scripts/1.upload_all.py")
        return False, None
    
    # Check code bucket has preprocessing script
    try:
        response = s3_client.head_object(Bucket=code_bucket, Key='preprocessing/preprocessing.py')
        print(f"Preprocessing script found: s3://{code_bucket}/preprocessing/preprocessing.py")
        print(f"   Size: {response['ContentLength']} bytes")
    except Exception as e:
        print(f"Preprocessing script not found: {e}")
        print("   Run: python scripts/1.upload_all.py")
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
    
    print("All processing prerequisites verified")
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
                
                return True
                
            elif status == 'Failed':
                print(f"\nProcessing job failed!")
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"Failure reason: {failure_reason}")
                
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
            return False
        else:
            print("All expected files found!")
        
        return True
        
    except Exception as e:
        print(f"Error verifying results: {e}")
        return False


# 2. Training Job Functions
def verify_training_prerequisites():
    """Verify all prerequisites for training job"""
    print("\nVerifying training prerequisites...")
    
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
        print("   Run: python scripts/1.upload_all.py")
        return False, None
    
    print("✓ All training prerequisites verified")
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
            return True
        else:
            print("⚠ Warning: model.tar.gz not found")
            return False
        
    except Exception as e:
        print(f"✗ Error verifying results: {e}")
        return False


# 3. Model Registration Functions
def get_latest_training_job():
    """Get the latest completed training job"""
    print("\nFinding latest completed training job...")
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        # List training jobs, sorted by creation time (newest first)
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
            print("No completed Titanic training jobs found")
            return None
        
        latest_job = titanic_jobs[0]
        job_name = latest_job['TrainingJobName']
        
        print(f"Latest training job: {job_name}")
        print(f"Status: {latest_job['TrainingJobStatus']}")
        print(f"End time: {latest_job['TrainingEndTime']}")
        
        return job_name
        
    except Exception as e:
        print(f"Error finding training job: {e}")
        return None


def get_training_job_details(job_name):
    """Get detailed information about the training job"""
    print(f"Getting details for training job: {job_name}")
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
        training_image = response['AlgorithmSpecification']['TrainingImage']
        role_arn = response['RoleArn']
        
        print(f"Model artifacts: {model_artifacts}")
        print(f"Training image: {training_image}")
        
        return {
            'model_artifacts': model_artifacts,
            'training_image': training_image,
            'role_arn': role_arn,
            'job_name': job_name
        }
        
    except Exception as e:
        print(f"Error getting training job details: {e}")
        return None


def create_model_package_group():
    """Create a model package group for Titanic models"""
    print("Creating model package group...")
    
    sagemaker_client = boto3.client('sagemaker')
    group_name = "titanic-survival-model-group"
    
    try:
        # Check if group already exists
        try:
            response = sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            print(f"Model package group already exists: {group_name}")
            return group_name
        except sagemaker_client.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise e
        
        # Create new group
        response = sagemaker_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Model package group for Titanic survival prediction models"
        )
        
        print(f"Created model package group: {group_name}")
        print(f"ARN: {response['ModelPackageGroupArn']}")
        
        return group_name
        
    except Exception as e:
        print(f"Error creating model package group: {e}")
        return None


def register_model_version(training_details, group_name):
    """Register a new model version"""
    print("Registering model version...")
    
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
        print(f"✓ Model registered successfully!")
        print(f"Model Package ARN: {model_package_arn}")
        
        return model_package_arn
        
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return None


def approve_model(model_package_arn):
    """Approve the model for deployment"""
    print("Approving model for deployment...")
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription="Model approved for deployment after successful training"
        )
        
        print("✓ Model approved for deployment!")
        return True
        
    except Exception as e:
        print(f"✗ Error approving model: {e}")
        return False


def main():
    """Main function to execute processing, training, and registration pipeline"""
    print("SageMaker ML Pipeline: Processing → Training → Registration")
    print("=" * 70)
    
    # Step 1: Processing Job
    print("\n1. PROCESSING JOB")
    print("=" * 20)
    
    # Verify processing prerequisites
    success, outputs = verify_processing_prerequisites()
    if not success:
        print("\n✗ Processing prerequisites not met. Please fix issues above.")
        return
    
    # Create and monitor processing job
    processing_job_name = create_processing_job(outputs)
    if not processing_job_name:
        print("\n✗ Failed to create processing job")
        return
    
    print(f"\nProcessing job '{processing_job_name}' is starting...")
    print("This may take 5-15 minutes depending on data size and complexity.")
    
    processing_success = monitor_processing_job(processing_job_name)
    
    if not processing_success:
        print("\n✗ Processing job failed. Cannot proceed to training.")
        return
    
    # Verify processing results
    if not verify_processing_results(outputs):
        print("\n✗ Processing completed but results verification failed. Cannot proceed to training.")
        return
    
    print("\n✓ Processing job completed successfully!")
    
    # Step 2: Training Job (only if processing succeeded)
    print("\n2. TRAINING JOB")
    print("=" * 15)
    
    # Verify training prerequisites
    training_success, outputs = verify_training_prerequisites()
    if not training_success:
        print("\n✗ Training prerequisites not met.")
        return
    
    # Create and monitor training job
    training_job_name = create_training_job(outputs)
    if not training_job_name:
        print("\n✗ Failed to create training job")
        return
    
    print(f"\nTraining job '{training_job_name}' is starting...")
    print("This may take 10-20 minutes depending on data size and model complexity.")
    
    training_success = monitor_training_job(training_job_name)
    
    if not training_success:
        print("\n✗ Training job failed. Cannot proceed to model registration.")
        return
    
    # Verify training results
    if not verify_training_results(outputs, training_job_name):
        print("\n✗ Training completed but results verification failed. Cannot proceed to registration.")
        return
    
    print("\n✓ Training job completed successfully!")
    
    # Step 3: Model Registration (only if training succeeded)
    print("\n3. MODEL REGISTRATION")
    print("=" * 20)
    
    # Get training job details
    training_details = get_training_job_details(training_job_name)
    if not training_details:
        print("\n✗ Could not get training job details for registration")
        return
    
    # Create model package group
    group_name = create_model_package_group()
    if not group_name:
        print("\n✗ Could not create model package group")
        return
    
    # Register model version
    model_package_arn = register_model_version(training_details, group_name)
    if not model_package_arn:
        print("\n✗ Could not register model")
        return
    
    # Approve model
    if approve_model(model_package_arn):
        print("✓ Model approval successful")
    else:
        print("⚠ Model registration successful but approval failed")
        print("You can approve manually in the SageMaker console")
    
    # Final Success Message
    print("\n" + "=" * 70)
    print("✓ COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Processing Job: {processing_job_name} - Completed")
    print(f"Training Job: {training_job_name} - Completed")
    print(f"Model Group: {group_name}")
    print(f"Model Package: {model_package_arn}")
    print("Status: Model trained and registered, ready for deployment")
    print("\nNext step: Deploy endpoint for inference")
    print("Command: python scripts/create_endpoint.py")


if __name__ == "__main__":
    main()