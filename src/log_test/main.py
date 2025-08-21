import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.logger import create_logger

logger = create_logger(name="log_test")
logger.info("This is an info message.")