# MLOps Energy Consumption - AWS EC2 MLflow Setup Guide

## Overview

This guide walks you through setting up a remote MLflow Tracking Server on AWS EC2 with PostgreSQL (Neon) backend and S3 artifact storage.

## Prerequisites

- AWS Account with EC2 and S3 access
- Neon PostgreSQL account (https://neon.com)
- SSH key pair for EC2 access

## Step 1: Create Neon PostgreSQL Database

1. Go to https://neon.com and sign in
2. Create a new project: `mlflow-tracking`
3. Create a database: `mlflow_db`
4. Note down the connection details:
   - Host: `your-project.neon.tech`
   - Database: `mlflow_db`
   - User: `your_user`
   - Password: `your_password`
   - Port: `5432`

## Step 2: Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://mlops-energy-artifacts --region us-east-1

# Set bucket policy (optional - for public read if needed)
aws s3api put-bucket-versioning \
    --bucket mlops-energy-artifacts \
    --versioning-configuration Status=Enabled
```

## Step 3: Launch EC2 Instance

1. **Launch Instance:**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t2.medium (2 vCPU, 4 GB RAM)
   - Storage: 20 GB gp3
   - Key Pair: Create or use existing

2. **Security Group Rules:**
   - SSH (22): Your IP
   - Custom TCP (5000): Your IP (for MLflow UI)
   - HTTPS (443): Anywhere (optional)

3. **Note the Public IP:** `your-ec2-public-ip`

## Step 4: Connect to EC2 and Install Dependencies

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install PostgreSQL client
sudo apt install postgresql-client -y

# Create virtual environment
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow and dependencies
pip install mlflow boto3 psycopg2-binary
```

## Step 5: Configure AWS Credentials on EC2

```bash
# Configure AWS CLI
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json
```

## Step 6: Start MLflow Tracking Server

```bash
# Set environment variables
export POSTGRES_USER="your_neon_user"
export POSTGRES_PASSWORD="your_neon_password"
export POSTGRES_HOST="your-project.neon.tech"
export POSTGRES_DB="mlflow_db"

# Build connection string
export BACKEND_STORE_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:5432/${POSTGRES_DB}"
export ARTIFACT_ROOT="s3://mlops-energy-artifacts"

# Start MLflow server
mlflow server \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
```

## Step 7: Run MLflow as a Service (Optional)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/mlflow.service
```

Add the following content:

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="POSTGRES_USER=your_neon_user"
Environment="POSTGRES_PASSWORD=your_neon_password"
Environment="POSTGRES_HOST=your-project.neon.tech"
Environment="POSTGRES_DB=mlflow_db"
ExecStart=/home/ubuntu/mlflow-env/bin/mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:5432/${POSTGRES_DB} \
    --default-artifact-root s3://mlops-energy-artifacts \
    --host 0.0.0.0 \
    --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

## Step 8: Configure Local Environment

Update your local `.env` file:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://your-ec2-public-ip:5000

# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=mlops-energy-artifacts

# Neon PostgreSQL
POSTGRES_USER=your_neon_user
POSTGRES_PASSWORD=your_neon_password
POSTGRES_HOST=your-project.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=mlflow_db
```

## Step 9: Test Connection

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://your-ec2-public-ip:5000")

# Test connection
print(mlflow.get_tracking_uri())
print(mlflow.list_experiments())
```

## Step 10: Access MLflow UI

Open your browser and navigate to:
```
http://your-ec2-public-ip:5000
```

## Troubleshooting

### Cannot connect to MLflow server
- Check EC2 security group allows port 5000 from your IP
- Verify MLflow service is running: `sudo systemctl status mlflow`
- Check logs: `sudo journalctl -u mlflow -f`

### S3 artifact upload fails
- Verify AWS credentials are configured correctly
- Check S3 bucket exists and has correct permissions
- Ensure EC2 instance has internet access

### PostgreSQL connection fails
- Verify Neon database is running
- Check connection string is correct
- Test connection: `psql "postgresql://user:pass@host:5432/db"`

## Security Best Practices

1. **Use HTTPS**: Set up SSL/TLS with Let's Encrypt
2. **Restrict Access**: Use VPN or IP whitelisting
3. **Rotate Credentials**: Regularly update AWS keys and database passwords
4. **Enable Logging**: Monitor access logs
5. **Backup Database**: Regular PostgreSQL backups

## Cost Optimization

- **EC2**: Use t2.micro for development (free tier eligible)
- **S3**: Enable lifecycle policies to archive old artifacts
- **Neon**: Use free tier for development
- **Stop EC2**: Stop instance when not in use

## Next Steps

1. Run model training scripts with MLflow tracking
2. Register models in MLflow Model Registry
3. Deploy models using MLflow Models
4. Set up monitoring and alerts
