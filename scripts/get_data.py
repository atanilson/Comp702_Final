"""
Downloads Data using URL
"""
import os

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
NAME = "EuroSAT.zip"
FOLDER = "data"

def download():
  os.system(f"wget {URL} -O {NAME}")
  os.system(f"unzip -q {NAME} -d {FOLDER}")
  os.remove(NAME)
