# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run the application
CMD ["python", "generate_fake.py"]