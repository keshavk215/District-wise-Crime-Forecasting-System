# AWS Infrastructure 
# This Terraform script sets up the necessary AWS resources for the Crime Forecasting System.

provider "aws" {
  region = "us-east-1"
}

# 1. S3 BUCKET (Model Storage)
resource "aws_s3_bucket" "model_bucket" {
  bucket = "crime-forecasting-models-prod"
}

# 2. ECR REPOSITORY (Docker Images)
resource "aws_ecr_repository" "crime_api_repo" {
  name = "crime-forecasting-api"
}

# 3. ECS CLUSTER (Container Orchestration)
resource "aws_ecs_cluster" "main" {
  name = "crime-forecasting-cluster"
}

# 4. RDS INSTANCE (PostgreSQL)
resource "aws_db_instance" "default" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.t3.micro"
  name                 = "crime_db"
  username             = "admin"
  password             = "securepassword"
  skip_final_snapshot  = true
}

# 5. API GATEWAY (Rate Limiting & Security)
resource "aws_apigatewayv2_api" "api_gateway" {
  name          = "crime_api_gateway"
  protocol_type = "HTTP"
}

# Define Rate Limiting (Throttling)
resource "aws_apigatewayv2_stage" "prod" {
  api_id = aws_apigatewayv2_api.api_gateway.id
  name   = "prod"
  
  default_route_settings {
    throttling_burst_limit = 50
    throttling_rate_limit  = 100
  }
}