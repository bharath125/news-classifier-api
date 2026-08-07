# Step 1 — base image (Python 3.12 already installed)
FROM python:3.12-slim

# Step 2 — set working directory inside container
WORKDIR /app

# Step 3 — copy requirements first (for caching)
COPY requirements.txt .

# Step 4 — install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 — copy your app code
COPY app.py .

# Step 6 — expose port
EXPOSE 5001

# Step 7 — run the app
CMD ["python", "app.py"]
