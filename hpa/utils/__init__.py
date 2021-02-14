import os


def create_folder(path):
    """Create a folder if it does not already exist"""
    if not (os.path.exists(path) or os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)
