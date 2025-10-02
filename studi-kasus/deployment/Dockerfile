# Base image Python 3.10.12 versi slim untuk ukuran lebih kecil
FROM python:3.10.12-slim-buster

# Set working directory di container
WORKDIR /app

# Copy requirements.txt terlebih dahulu untuk layer caching
COPY requirements.txt .

# Install dependencies tanpa cache untuk hemat storage
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi ke container
COPY . .

# Expose port 5000 untuk akses aplikasi
EXPOSE 5000

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]