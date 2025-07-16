import boto3
import time
from datetime import datetime

def cleanup_failed_resources():
    """Clean up the failed stack first"""
    print("Cleaning up failed resources...")
    
    try:
        # Get the failed endpoint name from CloudFormation events
        cf_client = boto3.client('cloudformation')
        
        # Try to delete the stack - it will clean up the failed endpoint
        try:
            cf_client.delete_stack(StackName='CdkStack')
            print("✓ Initiated stack deletion...")
            
            # Wait for deletion to complete
            waiter = cf_client.get_waiter('stack_delete_complete')
            print("Waiting for stack deletion to complete...")
            waiter.wait(StackName='CdkStack', WaiterConfig={'Delay': 30, 'MaxAttempts': 60})
            print("✓ Stack deleted successfully")
            
        except Exception as e:
            print(f"Stack deletion issue: {e}")
            # Continue anyway
        
    except Exception as e:
        print(f"Cleanup error: {e}")

def get_sagemaker_role():
    """Get SageMaker role from CDK stack or existing role"""
    try:
        # Try to get from CDK stack first
        cf_client = boto3.client('cloudformation')
        response = cf_client.describe_stacks(StackName='CdkStack')
        
        outputs = {}
        for output in response['Stacks'][0]['Outputs']:
            outputs[output['OutputKey']] = output['OutputValue']
        
        role_arn = outputs.get('SageMakerRoleArn')
        if role_arn:
            print(f"✓ Found SageMaker role from CDK: {role_arn}")
            return role_arn
            
    except Exception as e:
        print(f"Could not get role from CDK stack: {e}")
    
    # Try to find existing role
    try:
        iam_client = boto3.client('iam')
        role_name = 'SageMakerExecutionRole-us-east-1'
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        print(f"✓ Found existing role: {role_arn}")
        return role_arn
        
    except Exception as e:
        print(f"Could not find existing role: {e}")
        return None

def create_simple_endpoint():
    """Create endpoint directly using SageMaker Python SDK"""
    print("Creating simple endpoint using Python SDK...")
    
    # First, get the SageMaker role
    role_arn = get_sagemaker_role()
    if not role_arn:
        print("✗ No SageMaker role found. Please run 'cdk deploy' first.")
        return None
    
    sagemaker_client = boto3.client('sagemaker')
    
    # Get the latest training job
    training_jobs = sagemaker_client.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending', 
        StatusEquals='Completed',
        MaxResults=5
    )
    
    titanic_jobs = [job for job in training_jobs['TrainingJobSummaries'] 
                   if 'titanic' in job['TrainingJobName'].lower()]
    
    if not titanic_jobs:
        print("✗ No completed Titanic training jobs found")
        return None
    
    latest_job = titanic_jobs[0]['TrainingJobName']
    print(f"Using training job: {latest_job}")
    
    # Get job details
    job_details = sagemaker_client.describe_training_job(TrainingJobName=latest_job)
    model_artifacts = job_details['ModelArtifacts']['S3ModelArtifacts']
    training_image = job_details['AlgorithmSpecification']['TrainingImage']
    
    print(f"Model artifacts: {model_artifacts}")
    print(f"Training image: {training_image}")
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Step 1: Create Model (without Model Registry)
    model_name = f"titanic-simple-model-{timestamp}"
    
    try:
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': training_image,
                'ModelDataUrl': model_artifacts,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'train.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn=role_arn
        )
        print(f"✓ Model created: {model_name}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return None
    
    # Step 2: Create Endpoint Configuration
    config_name = f"titanic-simple-config-{timestamp}"
    
    try:
        config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium',
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"✓ Endpoint config created: {config_name}")
        
    except Exception as e:
        print(f"✗ Error creating endpoint config: {e}")
        return None
    
    # Step 3: Create Endpoint
    endpoint_name = f"titanic-simple-endpoint-{timestamp}"
    
    try:
        endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✓ Endpoint creation initiated: {endpoint_name}")
        
        return endpoint_name
        
    except Exception as e:
        print(f"✗ Error creating endpoint: {e}")
        return None

def wait_for_endpoint(endpoint_name):
    """Wait for endpoint to be ready"""
    print(f"Waiting for endpoint to be ready: {endpoint_name}")
    print("This may take 8-12 minutes...")
    
    sagemaker_client = boto3.client('sagemaker')
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            elapsed = int(time.time() - start_time)
            
            print(f"[{elapsed:3d}s] Status: {status}")
            
            if status == 'InService':
                print("✓ Endpoint is ready!")
                return True
            elif status == 'Failed':
                print("✗ Endpoint failed!")
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"Failure reason: {failure_reason}")
                return False
            
            time.sleep(30)
            
        except Exception as e:
            print(f"Error checking endpoint: {e}")
            return False

def test_simple_endpoint(endpoint_name):
    """Test the endpoint"""
    print(f"Testing endpoint: {endpoint_name}")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Simple test data - just the features as numbers
    test_cases = [
        {
            "name": "Rich First Class Woman",
            "data": [1, 0, 25, 0, 1, 100.0, 0, 0]
        },
        {
            "name": "Poor Third Class Man", 
            "data": [3, 1, 30, 0, 0, 7.25, 0, 1]
        }
    ]
    
    print("Making predictions:")
    print("-" * 40)
    
    for case in test_cases:
        try:
            # Try different payload formats
            import json
            
            # Format 1: Simple array
            payload = json.dumps([case["data"]])
            
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            
            print(f"🧑 {case['name']}:")
            print(f"   Input: {case['data']}")
            print(f"   Result: {result}")
            print()
            
        except Exception as e:
            print(f"✗ Error testing {case['name']}: {e}")
            print()

def main():
    """Main function"""
    print("Simple SageMaker Endpoint Creator")
    print("=" * 50)
    
    # Step 2: Create simple endpoint
    endpoint_name = create_simple_endpoint()
    if not endpoint_name:
        print("✗ Failed to create endpoint")
        return
    
    # Step 3: Wait for endpoint to be ready
    if not wait_for_endpoint(endpoint_name):
        print("✗ Endpoint failed to start")
        return
    
    # Step 4: Test endpoint
    test_simple_endpoint(endpoint_name)
    
    print("\n" + "=" * 50)
    print("✓ SIMPLE ENDPOINT COMPLETED!")
    print("=" * 50)
    print(f"Endpoint: {endpoint_name}")
    print("Status: Ready for inference")
    print(f"\nTo delete: aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")

if __name__ == "__main__":
    main()