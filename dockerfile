# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and assets
COPY src/ src/
COPY data/ data/

# Expose default Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "src/app.py"]
