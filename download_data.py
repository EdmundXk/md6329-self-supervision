"""Downloads the MedMNIST v2 64x64 archives used by this project."""
import os, urllib.request
os.makedirs("data", exist_ok=True)
for name in ["pneumoniamnist_64", "dermamnist_64"]:
    dest = f"data/{name}.npz"
    if os.path.exists(dest):
        print("have", dest); continue
    url = f"https://zenodo.org/records/10519652/files/{name}.npz?download=1"
    urllib.request.urlretrieve(url, dest)
    print("downloaded", dest, os.path.getsize(dest) // 1024, "KB")
