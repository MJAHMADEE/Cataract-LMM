# 🐳 Docker Configuration

## Overview

This directory contains Docker-related configurations and utilities for containerizing the Cataract-LMM framework, enabling consistent deployment across different environments.

## 📁 Contents

### **Files**

| File | Description |
|------|-------------|
| `healthcheck.py` | Health check script for Docker containers to monitor application status |
| `__init__.py` | Python package initialization file |

## 🚀 Docker Integration

### **Health Monitoring**

The `healthcheck.py` script provides:

- **Application Status Monitoring**: Checks if the surgical video processing services are running
- **Dependency Validation**: Verifies critical dependencies (CUDA, FFmpeg, Python packages)
- **Resource Availability**: Monitors GPU memory, disk space, and system resources
- **Service Endpoint Testing**: Validates API endpoints and model loading status

### **Usage in Dockerfile**

```dockerfile
# Add health check to your Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python docker/healthcheck.py
```

### **Container Health States**

- **✅ Healthy**: All services operational, models loaded, resources available
- **⚠️ Starting**: Container initializing, models loading
- **❌ Unhealthy**: Service failures, resource constraints, or dependency issues

## 🔧 Configuration

### **Environment Variables**

The health check script recognizes these environment variables:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
GPU_MEMORY_THRESHOLD=0.8

# Service Endpoints
API_HOST=localhost
API_PORT=8000
MODEL_ENDPOINT=/api/v1/models/status

# Health Check Intervals
HEALTH_CHECK_TIMEOUT=10
MAX_RETRY_ATTEMPTS=3
```

### **Custom Health Checks**

Extend the health check for specific requirements:

```python
from docker.healthcheck import BaseHealthCheck

class CustomHealthCheck(BaseHealthCheck):
    def check_surgical_models(self):
        # Custom model validation logic
        pass
        
    def check_video_processing(self):
        # Video pipeline health validation
        pass
```

## 🛠️ Development

### **Testing Health Checks**

```bash
# Run health check manually
python docker/healthcheck.py

# Test with specific configuration
CUDA_VISIBLE_DEVICES=0 python docker/healthcheck.py --verbose

# Check exit codes
echo $?  # 0 = healthy, 1 = unhealthy
```

### **Integration with Docker Compose**

```yaml
services:
  cataract-lmm:
    build: .
    healthcheck:
      test: ["CMD", "python", "docker/healthcheck.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

## 📊 Monitoring

### **Health Check Logs**

Monitor container health through Docker logs:

```bash
# View health check logs
docker logs --follow <container_name>

# Filter health check events
docker logs <container_name> 2>&1 | grep "HEALTH"
```

### **Integration with Orchestration**

- **Kubernetes**: Configure liveness and readiness probes
- **Docker Swarm**: Automatic service restart on health failures
- **Monitoring Systems**: Export health metrics to Prometheus/Grafana

## 🔍 Troubleshooting

### **Common Issues**

| Issue | Cause | Solution |
|-------|-------|----------|
| GPU Not Detected | CUDA not available | Check nvidia-docker runtime |
| Model Loading Timeout | Insufficient memory | Increase container memory limits |
| Service Unreachable | Port binding issues | Verify port mapping in docker run |

### **Debug Mode**

Enable verbose health checking:

```bash
# Run with debug output
python docker/healthcheck.py --debug --verbose
```

## 📚 Best Practices

1. **Resource Limits**: Set appropriate CPU/memory limits for containers
2. **Multi-stage Builds**: Use multi-stage Dockerfiles for smaller images
3. **Security Scanning**: Regularly scan images for vulnerabilities
4. **Health Check Frequency**: Balance monitoring needs with resource usage
5. **Graceful Shutdowns**: Implement proper signal handling for clean exits

## 🤝 Contributing

When adding new health checks:

1. Extend the `BaseHealthCheck` class
2. Add appropriate timeout handling
3. Include meaningful error messages
4. Test with various failure scenarios
5. Update documentation with new check types

---

*This Docker configuration ensures reliable, monitorable deployments of the Cataract-LMM framework in production environments.*
