# Official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and essential files (setup.py, README.md) first
COPY requirements.txt setup.py README.md ./

# Install dependencies (this will install your package in editable mode via "-e .")
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with S3 support (adjust if needed)
RUN pip install dvc[s3]

# Copy the rest of your project code (all files and directories)
COPY . .

# Expose ports (FastAPI on 8000 and Streamlit on 8501)
EXPOSE 8000
EXPOSE 8501

# Command to run FastAPI (using Uvicorn) and Streamlit concurrently
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8501 & streamlit run streamlit_app.py --server.port=8000 --server.address=0.0.0.0"]
