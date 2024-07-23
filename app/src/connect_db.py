import sqlite3
from pathlib import Path
import sys
sys.path.append('./app/src')
from config import config
from langchain_community.utilities import SQLDatabase


db_path = config.ARTIFACTS_PATH / 'sql_database.db'
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')
print(db.get_usable_table_names())
