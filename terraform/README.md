# Terraform — AWS Deployment

## Resources Created

| Resource | Type | Description |
|---|---|---|
| EC2 instance | `t3.medium` | Runs the full Docker stack |
| S3 bucket | `picnic-ml-platform-artifacts` | Versioned model artifact storage |
| Security group | — | Opens ports 8000, 5000, 22 |
| IAM role | — | EC2 → S3 read/write access |

## Usage

```bash
cd terraform/

# 1. Initialise providers
terraform init

# 2. Preview changes
terraform plan -var="key_pair_name=your-key"

# 3. Apply
terraform apply -var="key_pair_name=your-key"

# 4. Get outputs
terraform output api_url
terraform output mlflow_url

# 5. Destroy when done
terraform destroy
```

## Variables

| Variable | Default | Description |
|---|---|---|
| `region` | `eu-west-1` | AWS region |
| `instance_type` | `t3.medium` | EC2 size |
| `bucket_name` | `picnic-ml-platform-artifacts` | S3 bucket name (must be globally unique) |
| `key_pair_name` | `""` | Existing EC2 key pair for SSH |
