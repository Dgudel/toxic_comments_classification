# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable for Flask to run on 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0

# Run chatapp.py when the container launches
CMD ["python", "chatapp.py"]



