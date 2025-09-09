#!/bin/bash

# AWS Deployment Script for RAG Application
# Run this on your EC2 instance after SSH connection

set -e

echo "Starting AWS deployment of RAG application..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "Installing Docker..."
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git if not present
sudo apt install git -y

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Logout and login again: exit"
echo "2. Clone your repository: git clone <your-repo-url>"
echo "3. Set up environment: cp env.example .env && nano .env"
echo "4. Deploy: docker-compose up -d --build"
echo ""
echo "Your application will be available at:"
echo "  - Streamlit UI: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
echo "  - API Docs: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/docs"
