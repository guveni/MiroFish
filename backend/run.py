"""MiroFish Backend Entrypoint using FastAPI and Uvicorn"""

import os
import sys

# Fix for Windows console encoding issues: set UTF-8 encoding before all imports
if sys.platform == 'win32':
    # 设置环境变量确保 Python 使用 UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # 重新配置标准输出流为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config

# Create global app instance for ASGI servers (Uvicorn, etc.)
app = create_app()


def main():
    """Main function"""
    # 验证配置
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\n请检查 .env 文件中的配置")
        sys.exit(1)
    
    # Get run configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # Start the service with Uvicorn
    import uvicorn
    uvicorn.run(
        "run:app", 
        host=host, 
        port=port, 
        reload=debug
    )


if __name__ == '__main__':
    main()
