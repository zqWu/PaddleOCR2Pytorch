import os.path
from os import path

PROJECT_ROOT_DIR = path.dirname(path.abspath(__file__))


# 这个文件必须放在项目根目录

# print(PROJECT_ROOT_DIR)

def relative_path_in_root(rel_path_in_root: str):
    """根据相对路径, 得到 dir """
    return os.path.join(PROJECT_ROOT_DIR, rel_path_in_root)
