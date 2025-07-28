# Use a specific Python version for consistency and compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Pre-install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files including models and scripts
COPY . .

# Accept collection directory (like Collection_1) as runtime arg
ENTRYPOINT ["python", "run_analysis.py"]
