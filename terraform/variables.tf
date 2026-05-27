variable "region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-1"
}

variable "instance_type" {
  description = "EC2 instance type for the ML platform server"
  type        = string
  default     = "t3.medium"
}

variable "bucket_name" {
  description = "S3 bucket name for model artifact storage"
  type        = string
  default     = "picnic-ml-platform-artifacts"
}

variable "project_name" {
  description = "Project prefix applied to all resource names and tags"
  type        = string
  default     = "picnic-ml-platform"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access (create in AWS console first)"
  type        = string
  default     = ""   # leave empty to skip SSH key association
}
