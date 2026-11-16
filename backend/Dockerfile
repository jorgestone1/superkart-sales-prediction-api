# Python 3.12 matches the environment the model was trained and serialized in
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first so this layer is cached when only the app code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and the serialized model
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Serve the Flask app with gunicorn: 2 worker processes (one per core on a 2-core Codespace), bound to port 7860
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
