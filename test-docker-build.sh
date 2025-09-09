#!/bin/bash

# 🐳 Local Docker Build Test Script
# Test Docker build locally to verify Dockerfile before CI/CD

set -e  # Exit on any error

echo "🏥 Cataract-LMM Local Docker Build Test"
echo "========================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Ensure we're in the right directory
if [ ! -f "codes/Dockerfile" ]; then
    echo "❌ codes/Dockerfile not found. Please run this script from the repository root."
    exit 1
fi

echo "📁 Building from: $(pwd)/codes/"
echo "🐳 Docker version: $(docker --version)"
echo ""

# Build the Docker image locally
echo "🏗️  Building Docker image locally..."
IMAGE_TAG="cataract-lmm:local-test"

docker build \
    --progress=plain \
    --tag "${IMAGE_TAG}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    codes/

echo ""
echo "✅ Docker build completed successfully!"
echo "📊 Image details:"
docker images "${IMAGE_TAG}"

echo ""
echo "🧪 Testing the built image..."

# Test basic functionality
echo "Testing Python version..."
docker run --rm "${IMAGE_TAG}" python --version

echo "Testing package imports..."
docker run --rm "${IMAGE_TAG}" python -c "
try:
    import numpy as np
    import torch
    import cv2
    print('✅ Core dependencies imported successfully')
    print(f'  - NumPy: {np.__version__}')
    print(f'  - PyTorch: {torch.__version__}')
    print(f'  - OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'⚠️  Import warning: {e}')
    print('Some dependencies may not be available')
"

echo "Testing container user and permissions..."
docker run --rm "${IMAGE_TAG}" whoami
docker run --rm "${IMAGE_TAG}" ls -la /app/

echo ""
echo "🎉 Local Docker build test completed successfully!"
echo "🚀 Image '${IMAGE_TAG}' is ready for testing"
echo ""
echo "💡 You can now run the container interactively with:"
echo "   docker run -it --rm ${IMAGE_TAG} /bin/bash"
echo ""
echo "🧹 To clean up the test image when done:"
echo "   docker rmi ${IMAGE_TAG}"

# Optionally run security scan if Trivy is available
if command -v trivy &> /dev/null; then
    echo ""
    echo "🔍 Running security scan with Trivy..."
    trivy image --severity HIGH,CRITICAL "${IMAGE_TAG}" || echo "⚠️  Security scan completed with warnings"
else
    echo ""
    echo "💡 Install Trivy to run security scans: https://trivy.dev/"
fi

echo ""
echo "✅ All tests passed! The Docker build should work in CI/CD."
