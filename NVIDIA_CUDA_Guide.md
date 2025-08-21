# Local AI/ML Development Environment Setup Guide
#### Guide current as of 08/20/2025

## Prerequisites Check
- ✅ Install WSL2 (if using Windows)
- ✅ NVIDIA RTX GPU (tested with RTX 4070)
- ✅ Docker Desktop for Windows
- ✅ Windows 11 or Windows 10 version 21H2+ 
- ✅ 16GB+ RAM recommended
- ✅ Plenty of storage space for models

## Step 1: Install NVIDIA Container Toolkit (For Docker GPU Access)
### The following commands should be run from your linux/WSL terminal (not Windows command prompt)

### Windows with WSL2:
```bash
# In WSL2 Ubuntu terminal - Run each command separately

# First, clean up any existing problematic sources
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Check your distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "Distribution: $distribution"

# Add the GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the repository with correct format
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker Desktop from Windows (NOT from WSL2)
echo "🔄 Please restart Docker Desktop from Windows now..."
echo "Right-click Docker Desktop icon → Restart"
echo "Wait for Docker to fully start, then continue..."

# Verify GPU access works
echo "Testing GPU access..."
docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

**Expected Output:** You should see your GPU detected and performance metrics like "billion interactions per second"

## Useful Commands

```bash
# Monitor resource usage
docker stats

# View logs
docker-compose logs ollama
docker-compose logs jupyter

# Stop all services
docker-compose down

# Update images
docker-compose pull
docker-compose up -d

# Shell into containers
docker exec -it ollama bash
docker exec -it ai-jupyter bash

# Check GPU usage
nvidia-smi
```

## GPU Optimization Tips

**For RTX 4070 (12GB VRAM):**
- Use 7B parameter models for optimal performance
- Use quantized models (Q4_K_M, Q5_K_M) for memory efficiency
- Monitor GPU memory with `nvidia-smi`
- Context windows: 2048-4096 tokens recommended

**Performance Expectations:**
- **GPU**: 2-5 seconds per response
- **CPU**: 30-60 seconds per response

## Troubleshooting

**GPU not detected in Docker:**
- Ensure latest NVIDIA Windows drivers installed
- Restart Docker Desktop after installing NVIDIA Container Toolkit
- Check Windows version is 21H2 or newer

**Out of memory errors:**
- Use smaller models (3B instead of 7B)
- Reduce context window size
- Close other GPU-using applications

**Slow inference despite GPU:**
- Verify GPU is being used: `docker logs ollama`
- Check `nvidia-smi` for GPU utilization
- Ensure Docker Desktop has WSL2 integration enabled

**Container build failures:**
- Check internet connection for package downloads
- Try removing version constraints from pip installs
- Clear Docker cache: `docker system prune -f`

**Package import errors:**
- Install missing packages in Jupyter: `!pip install package-name`
- Rebuild containers: `docker-compose up -d --build`

## Advanced Configuration

**Enable more GPU memory for larger models:**
```yaml
# In docker-compose.yml, add to ollama service:
environment:
  - OLLAMA_GPU_MEMORY_FRACTION=0.9
```

**Custom model directory:**
```yaml
# Mount custom model path:
volumes:
  - /path/to/your/models:/root/.ollama
```

Please keep in mind that these instructions are specifically for the NVIDIA RTX 4070, however I have also successfully tested it with an NVIDIA RTX 3060.
If you run into any issues, I recommend checking NVIDIA's documentation.