#!/bin/bash
# EC2 Setup Script for Hallmark OCR API
# Run this on a fresh Ubuntu 22.04+ EC2 instance

set -e

echo "=== Hallmark OCR API - EC2 Setup ==="

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.10+ and dependencies
sudo apt-get install -y python3.10 python3.10-venv python3-pip
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Create app directory
sudo mkdir -p /opt/hallmark-ocr
sudo chown ubuntu:ubuntu /opt/hallmark-ocr

# Clone or copy your code to /opt/hallmark-ocr
# cd /opt/hallmark-ocr
# git clone https://github.com/yourusername/ocr-detection.git .

# Create virtual environment
cd /opt/hallmark-ocr
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file (copy from template)
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please update with your credentials"
fi

# Create systemd service for API
sudo tee /etc/systemd/system/hallmark-api.service > /dev/null <<EOF
[Unit]
Description=Hallmark OCR API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hallmark-ocr
Environment=PATH=/opt/hallmark-ocr/venv/bin
ExecStart=/opt/hallmark-ocr/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for Dashboard
sudo tee /etc/systemd/system/hallmark-dashboard.service > /dev/null <<EOF
[Unit]
Description=Hallmark QC Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hallmark-ocr
Environment=PATH=/opt/hallmark-ocr/venv/bin
Environment=API_BASE_URL=http://localhost:8000
ExecStart=/opt/hallmark-ocr/venv/bin/streamlit run src/qc_dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable hallmark-api hallmark-dashboard
sudo systemctl start hallmark-api hallmark-dashboard

echo "=== Services started ==="
echo "API running on port 8000"
echo "Dashboard running on port 8501"
echo ""
echo "To check status: sudo systemctl status hallmark-api"
echo "To view logs: sudo journalctl -u hallmark-api -f"
