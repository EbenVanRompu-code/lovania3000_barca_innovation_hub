"""Download the DFL/IDSSE open dataset from Figshare into external_data/idsse/ (gitignored).

    uv run python -m possession.external.download

Dataset: "An integrated dataset of spatiotemporal and event data in elite soccer",
doi 10.6084/m9.figshare.28196177, licence CC BY 4.0 (see possession/external/LICENSES.md).
Files are verified against the md5 that Figshare publishes.
"""

import hashlib
import json
import os
import urllib.request

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_data", "idsse")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=60) as r:
        article = json.load(r)
    print("licence:", article["license"]["name"], article["license"]["url"])
    for f in article["files"]:
        path = os.path.join(DEST, f["name"])
        if os.path.exists(path) and md5(path) == f["computed_md5"]:
            print("ok (cached)", f["name"])
            continue
        urllib.request.urlretrieve(f["download_url"], path)
        status = "ok" if md5(path) == f["computed_md5"] else "MD5 MISMATCH"
        print(status, f["name"])


if __name__ == "__main__":
    main()
