# Use Ubuntu as base image
FROM ubuntu:20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libboost-all-dev \
    libopencv-dev \
    libtbb-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dlib (required by OpenFace)
WORKDIR /opt
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    make install && \
    ldconfig

# Install OpenFace
WORKDIR /opt
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git && \
    cd OpenFace && \
    bash download_models.sh && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# Set OpenFace path
ENV OPENFACE_DIR=/opt/OpenFace

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "600", "--workers", "1", "backend.app:app"]
