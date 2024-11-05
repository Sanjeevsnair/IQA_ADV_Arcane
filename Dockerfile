# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the environment variable to set Streamlit to run on the main host IP
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_ENABLE_CORS false

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
