# Use the official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for required packages
RUN apt-get update && \
    apt-get install -y libpoppler-cpp-dev tesseract-ocr libgl1-mesa-glx && \
    apt-get clean

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (if needed for tokenization)
RUN python -m nltk.downloader punkt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
