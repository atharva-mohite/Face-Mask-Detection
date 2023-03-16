# Use TensorFlow base image
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Copy requirements file to container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the Flask app code to container
COPY app.py .

# Copy the HTML templates and static files to container
COPY templates/ templates/

# Expose the port that the app will run on
EXPOSE 5000

# Use gunicorn as the server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]