output "ec2_public_ip" {
  description = "Public IP of the ML platform EC2 instance"
  value       = aws_instance.ml_platform.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.ml_platform.public_dns
}

output "api_url" {
  description = "FastAPI inference server URL"
  value       = "http://${aws_instance.ml_platform.public_ip}:8000"
}

output "mlflow_url" {
  description = "MLflow experiment tracking UI URL"
  value       = "http://${aws_instance.ml_platform.public_ip}:5000"
}

output "s3_bucket_name" {
  description = "S3 bucket for model artifact storage"
  value       = aws_s3_bucket.artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 artifact bucket"
  value       = aws_s3_bucket.artifacts.arn
}
