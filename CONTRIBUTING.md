# Contributing to Multi-Agent AI Research System

We welcome contributions! Here's how you can help:

## Ways to Contribute

- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements
- 🔧 Add support for new AI frameworks
- 📝 Improve documentation
- 🎨 Enhance the user interface
- 🐳 Improve Docker configuration

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- Git for version control
- Basic knowledge of Python and AI frameworks

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/multi-agent-ai-research-system.git`
3. Copy environment file: `cp .env.example .env`
4. Start services: `docker-compose up`
5. Access Jupyter: http://localhost:8888
6. Access Streamlit: http://localhost:8501

### Making Changes
1. Create a new branch for your feature
2. Make your changes in the appropriate files
3. Test with Docker: `docker-compose down && docker-compose up --build`
4. Ensure all services work correctly
5. Submit a pull request

## Code Standards

- Follow PEP 8 Python style guidelines
- Add comments for complex logic
- Include docstrings for functions
- Test your changes in the Docker environment
- Update documentation for new features

## Docker Development

### Building Individual Services

# Rebuild Jupyter container
docker-compose build jupyter

# Rebuild Streamlit container  
docker-compose build streamlit

# Restart specific service
docker-compose restart ollama

### Logs and Debugging

# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs streamlit

# Interactive shell in container
docker-compose exec jupyter bash

## Questions?

Feel free to open an issue for any questions or discussions!