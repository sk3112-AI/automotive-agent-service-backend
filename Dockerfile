# Use a slim Python image as the base
# Using bookworm ensures it's based on Debian 12, a recent and stable Linux distribution.
FROM python:3.13-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# Assuming requirements.txt is in the root of your service folder (C:\Users\smkar\aoe-motors-agentservice\)
COPY requirements.txt .

# Install system dependencies (e.g., for pandas, zoneinfo/tzdata)
# tzdata provides timezone information, often needed for Python's zoneinfo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
# --no-cache-dir helps keep the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
# Assuming your main application file is automotive_agent_service.py
COPY automotive_agent_service.py .

# Expose the port your FastAPI app will run on
EXPOSE 10000

# Set the command to run your FastAPI application using Uvicorn
# The $PORT environment variable is provided by Render automatically
CMD ["uvicorn", "automotive_agent_service:app", "--host", "0.0.0.0", "--port", "10000"]