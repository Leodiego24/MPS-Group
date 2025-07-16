from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    RemovalPolicy,
    CfnOutput,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
    CfnParameter,
    Fn,
    CfnCondition,
)
from constructs import Construct

class CdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # =============================================================================
        # PARAMETERS (OPTIONAL FOR ENDPOINT DEPLOYMENT)
        # =============================================================================
        
        self.model_package_arn_param = CfnParameter(
            self, "ModelPackageArn",
            type="String",
            default="",
            description="ARN of the approved model package for endpoint deployment (optional)"
        )

        # =============================================================================
        # S3 BUCKET FOR DATA INGESTION (RESTRICTED ACCESS)
        # =============================================================================
        
        self.ingestion_bucket = s3.Bucket(
            self, "BucketIngestion",
            bucket_name=f"ml-ingestion-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True
        )

        # =============================================================================
        # SAGEMAKER ROLE WITH READ-ONLY PERMISSIONS TO BUCKET
        # =============================================================================
        
        self.sagemaker_execution_role = iam.Role(
            self, "SageMakerExecutionRole",
            role_name=f"SageMakerExecutionRole-{self.region}",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ]
        )

        # =============================================================================
        # SPECIFIC READ-ONLY POLICY FOR DATASET BUCKET
        # =============================================================================
        
        ingestion_read_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetBucketVersioning",
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:GetObjectTagging"
            ],
            resources=[
                self.ingestion_bucket.bucket_arn,
                f"{self.ingestion_bucket.bucket_arn}/*"
            ]
        )

        self.sagemaker_execution_role.add_to_policy(ingestion_read_policy)

        # =============================================================================
        # POLICIES FOR OTHER BUCKETS (PROCESSED DATA, MODELS)
        # =============================================================================
        
        self.processed_data_bucket = s3.Bucket(
            self, "ProcessedDataBucket",
            bucket_name=f"ml-processed-data-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED
        )

        self.model_artifacts_bucket = s3.Bucket(
            self, "ModelArtifactsBucket",
            bucket_name=f"ml-model-artifacts-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED
        )
        
        self.code_bucket = s3.Bucket(
            self, "CodeBucket",
            bucket_name=f"ml-code-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED
        )

        # Grant read/write permissions to working buckets
        self.processed_data_bucket.grant_read_write(self.sagemaker_execution_role)
        self.model_artifacts_bucket.grant_read_write(self.sagemaker_execution_role)
        self.code_bucket.grant_read(self.sagemaker_execution_role)

        # =============================================================================
        # ADDITIONAL CLOUDWATCH LOGS POLICY
        # =============================================================================
        logs_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams"
            ],
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*:*"
            ]
        )
        self.sagemaker_execution_role.add_to_policy(logs_policy)

        # =============================================================================
        # SAGEMAKER ENDPOINT RESOURCES (CONDITIONAL)
        # =============================================================================
        
        # Create endpoint resources only if model package ARN is provided
        # self.endpoint_resources = self._create_endpoint_resources()

        # =============================================================================
        # OUTPUTS FOR REFERENCE
        # =============================================================================
        
        CfnOutput(
            self, "IngestionBucketName",
            value=self.ingestion_bucket.bucket_name,
            description="Bucket with ingestion data (read-only for SageMaker)",
            export_name="IngestionBucketName"
        )

        CfnOutput(
            self, "ProcessedDataBucketName",
            value=self.processed_data_bucket.bucket_name,
            description="Bucket for processed data",
            export_name="ProcessedDataBucketName"
        )

        CfnOutput(
            self, "ModelArtifactsBucketName",
            value=self.model_artifacts_bucket.bucket_name,
            description="Bucket for model artifacts",
            export_name="ModelArtifactsBucketName"
        )

        CfnOutput(
            self, "SageMakerRoleArn",
            value=self.sagemaker_execution_role.role_arn,
            description="SageMaker role ARN with specific permissions",
            export_name="SageMakerRoleArn"
        )

        CfnOutput(
            self, "CodeBucketName",
            value=self.code_bucket.bucket_name,
            description="Bucket for ML scripts and code",
            export_name="CodeBucketName"
        )

    def _create_endpoint_resources(self):
        """Create SageMaker endpoint resources conditionally"""
        
        # Condition to check if model package ARN is provided
        model_arn_condition = CfnCondition(
            self, "ModelArnProvided",
            expression=Fn.condition_not(Fn.condition_equals(self.model_package_arn_param.value_as_string, ""))
        )
        
        # Generate timestamp for unique naming
        import time
        timestamp = str(int(time.time()))
        
        # 1. SageMaker Model
        model = sagemaker.CfnModel(
            self, "TitanicModel",
            model_name=f"titanic-model-{timestamp}",
            execution_role_arn=self.sagemaker_execution_role.role_arn,
            containers=[
                sagemaker.CfnModel.ContainerDefinitionProperty(
                    model_package_name=self.model_package_arn_param.value_as_string
                )
            ],
            tags=[
                {"key": "Project", "value": "TitanicMLPipeline"},
                {"key": "Environment", "value": "Production"},
                {"key": "ManagedBy", "value": "CDK"}
            ]
        )
        
        # Apply condition
        model.cfn_options.condition = model_arn_condition
        
        # 2. Endpoint Configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "TitanicEndpointConfig",
            endpoint_config_name=f"titanic-endpoint-config-{timestamp}",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="Primary",
                    model_name=model.model_name,
                    initial_instance_count=1,
                    instance_type="ml.t2.medium",
                    initial_variant_weight=1.0
                )
            ],
            tags=[
                {"key": "Project", "value": "TitanicMLPipeline"},
                {"key": "CostCenter", "value": "ML-Inference"},
                {"key": "Environment", "value": "Production"}
            ]
        )
        
        # Apply condition and dependency
        endpoint_config.cfn_options.condition = model_arn_condition
        endpoint_config.add_dependency(model)
        
        # 3. Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "TitanicEndpoint",
            endpoint_name=f"titanic-endpoint-{timestamp}",
            endpoint_config_name=endpoint_config.endpoint_config_name,
            tags=[
                {"key": "Project", "value": "TitanicMLPipeline"},
                {"key": "Environment", "value": "Production"},
                {"key": "AutoDelete", "value": "true"},
                {"key": "ManagedBy", "value": "CDK"}
            ]
        )
        
        # Apply condition and dependency
        endpoint.cfn_options.condition = model_arn_condition
        endpoint.add_dependency(endpoint_config)
        
        # Conditional Outputs for Endpoint
        CfnOutput(
            self, "ModelName",
            value=model.model_name,
            description="SageMaker Model Name",
            export_name="TitanicModelName",
            condition=model_arn_condition
        )
        
        CfnOutput(
            self, "EndpointConfigName", 
            value=endpoint_config.endpoint_config_name,
            description="SageMaker Endpoint Configuration Name",
            export_name="TitanicEndpointConfigName",
            condition=model_arn_condition
        )
        
        CfnOutput(
            self, "EndpointName",
            value=endpoint.endpoint_name,
            description="SageMaker Endpoint Name for Inference",
            export_name="TitanicEndpointName",
            condition=model_arn_condition
        )
        
        CfnOutput(
            self, "EndpointArn",
            value=endpoint.ref,
            description="SageMaker Endpoint ARN",
            export_name="TitanicEndpointArn", 
            condition=model_arn_condition
        )
        
        CfnOutput(
            self, "InferenceInstructions",
            value=f"Use: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint.endpoint_name} --content-type application/json --body '[sample_data]' output.json",
            description="CLI command to invoke the endpoint",
            condition=model_arn_condition
        )
        
        return {
            'model': model,
            'endpoint_config': endpoint_config,
            'endpoint': endpoint
        }