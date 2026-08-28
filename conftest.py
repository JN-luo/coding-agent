"""pytest 根配置：把仓库根目录加入 sys.path，保证 `import agent` 可用。

后续 sandbox/tools 的测试 fixture（如临时 workspace）也放在这里。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
