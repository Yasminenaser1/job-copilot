# Start from an official slim Python image
FROM python:3.13-slim

# Work inside /app in the container
WORKDIR /app

# Copy just requirements first (Docker caches this layer -
# rebuilds are fast when only code changes, not dependencies)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual code
COPY *.py ./
COPY evals/ evals/

# The API listens on 8000 inside the container
EXPOSE 8000

# What runs when the container starts
# --host 0.0.0.0 = listen inside the container (Docker maps it to your Mac)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
