import sys
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv(_root / ".env")
