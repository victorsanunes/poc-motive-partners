import os

from dotenv import load_dotenv
from envyaml import EnvYAML
from pathlib import Path

_current_dir = os.path.dirname(__file__)

load_dotenv(os.path.join(_current_dir, "../../../.env"))
CONFIG = EnvYAML(os.path.join(_current_dir, "config.yaml"))

# Directories
SRC_PATH = Path(__file__).resolve().parent
ROOT_PATH = SRC_PATH.parent.parent.parent
# DATA PATHS
DATA_PATH = ROOT_PATH / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'

# Create paths if nos exitst
# for p in [
#     DATA_PATH, 
#     RAW_DATA_PATH, 
#     PROCESSED_DATA_PATH, 
#     ]:
#     p.mkdir(exist_ok=True)