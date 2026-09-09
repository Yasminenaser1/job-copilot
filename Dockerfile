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

# Data the app needs at runtime: resume/profile, the Chroma index,
# saved postings, and the web frontend
COPY profile/ profile/
COPY db/ db/
COPY jobs/ jobs/
COPY sources/ sources/
COPY frontend/ frontend/

# Reach Ollama on the host Mac, not inside the container
ENV OLLAMA_HOST=http://host.docker.internal:11434

# The API listens on 8000 inside the container
EXPOSE 8000

# What runs when the container starts
# --host 0.0.0.0 = listen inside the container (Docker maps it to your Mac)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
