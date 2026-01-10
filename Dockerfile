# Use Ubuntu as base image
FROM ubuntu:20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =========================
# 1. System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 2. Install Modern CMake (>= 3.17)
# =========================
RUN apt-get remove -y cmake || true && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh && \
    chmod +x cmake-3.27.9-linux-x86_64.sh && \
    ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.27.9-linux-x86_64.sh

# Verify CMake version (optional but useful)
RUN cmake --version

# =========================
# 3. Build & Install dlib
# =========================
WORKDIR /opt
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# =========================
# 4. Build OpenFace
# =========================
WORKDIR /opt
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git && \
    cd OpenFace && \
    bash download_models.sh && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# =========================
# 5. Environment Variables
# =========================
ENV OPENFACE_DIR=/opt/OpenFace

# =========================
# 6. Application Setup
# =========================
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# =========================
# 7. Runtime
# =========================
EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "600", "--workers", "1", "backend.app:app"]
