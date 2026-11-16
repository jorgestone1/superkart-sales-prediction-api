# Minimal Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first so this layer is cached when only the app code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . .

# Streamlit serves on port 8501
EXPOSE 8501

# Run Streamlit on all interfaces; XSRF protection is disabled so CSV uploads work through the Codespaces port forward
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
