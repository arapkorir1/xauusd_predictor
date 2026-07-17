# Step 1: Use an official lightweight Python runtime based on Debian Slim
FROM python:3.10-slim

# Step 2: Set essential environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Keeps logs flowing in real-time to stdout without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Step 3: Set our working directory inside the container
WORKDIR /app

# Step 4: Install minimal system utilities needed to build scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Step 6: Upgrade pip and install application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 7: Copy our configuration, models, and application codebase into the image
COPY config/ /app/config/
COPY models/ /app/models/
COPY app/ /app/app/
COPY src/ /app/src/

# Step 8: Expose the network port our API will run on
EXPOSE 8000

# Step 9: Start our web application using Uvicorn
# We bind to 0.0.0.0 to allow external network traffic to access the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
