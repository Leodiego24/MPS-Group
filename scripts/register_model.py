import boto3
import json
from datetime import datetime
import time

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

def get_latest_training_job():
    """Get the latest completed training job"""
    print("Finding latest completed training job...")
    
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

def verify_model_registration(group_name):
    """Verify the model was registered correctly"""
    print("\nVerifying model registration...")
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        # List model packages in the group
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=group_name,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        if not response['ModelPackageSummaryList']:
            print("✗ No model packages found")
            return False
        
        print("✓ Registered models:")
        for model in response['ModelPackageSummaryList']:
            print(f"  Version: {model['ModelPackageVersion']}")
            print(f"  Status: {model['ModelPackageStatus']}")
            print(f"  Approval: {model['ModelApprovalStatus']}")
            print(f"  ARN: {model['ModelPackageArn']}")
            print(f"  Created: {model['CreationTime']}")
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying registration: {e}")
        return False

def main():
    """Main function to register model"""
    print("SageMaker Model Registry")
    print("=" * 40)
    
    # Step 1: Get stack outputs
    outputs = get_stack_outputs()
    if not outputs:
        print("✗ Could not get stack outputs")
        return
    
    # Step 2: Find latest training job
    job_name = get_latest_training_job()
    if not job_name:
        print("✗ No completed training job found")
        print("Run the training job first: python create_training_job_fixed.py")
        return
    
    # Step 3: Get training job details
    training_details = get_training_job_details(job_name)
    if not training_details:
        print("✗ Could not get training job details")
        return
    
    # Step 4: Create model package group
    group_name = create_model_package_group()
    if not group_name:
        print("✗ Could not create model package group")
        return
    
    # Step 5: Register model version
    model_package_arn = register_model_version(training_details, group_name)
    if not model_package_arn:
        print("✗ Could not register model")
        return
    
    # Step 6: Approve model
    if approve_model(model_package_arn):
        print("✓ Model approval successful")
    else:
        print("⚠ Model registration successful but approval failed")
        print("You can approve manually in the SageMaker console")
    
    # Step 7: Verify registration
    if verify_model_registration(group_name):
        print("\n" + "=" * 40)
        print("✓ MODEL REGISTRATION COMPLETED!")
        print("=" * 40)
        print(f"Model Group: {group_name}")
        print(f"Model Package: {model_package_arn}")
        print("Status: Approved and ready for deployment")
        print("\nNext step: Deploy endpoint for inference")
        print("Command: python deploy_endpoint.py")
    else:
        print("\n⚠ Registration completed but verification failed")

if __name__ == "__main__":
    main()