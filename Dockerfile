# Use a lightweight Python base
FROM python:3.9-slim

# Don't buffer stdout (logs appear immediately in CloudWatch)
ENV PYTHONUNBUFFERED=TRUE

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
# RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY models/ /app/models/ 

# Sagemaker specific: 
# SageMaker looks for an executable specifically named 'serve' in the path for inference
# We create a script that runs uvicorn
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

EXPOSE 8080

ENTRYPOINT ["entrypoint"]