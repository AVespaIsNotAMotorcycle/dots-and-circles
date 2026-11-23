'''
Downloads the pages users have submitted from the server's store (see
server/submissions.py) into ocr/submitted_data/ (gitignored), one folder
per submission, each with its submission.json and, for an upload, its
page.png. Submissions already downloaded are skipped, and nothing is
deleted from the store.

From the repository root, with a read-only R2 token in the environment:

  R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    R2_BUCKET=... .venv/bin/python -m ocr.fetch_submissions

That needs boto3 (`.venv/bin/pip install boto3`). --folder reads a local
store instead, as the server keeps with SUBMISSIONS_FOLDER.
'''

import argparse
import json
import os

from server.submissions import (JSON_NAME, PAGE_NAME, PREFIX, FolderStore,
                                is_id, store_from_env)

directory = os.path.dirname(os.path.abspath(__file__))
OUT_FOLDER = os.path.join(directory, "submitted_data")


def complete_ids(store):
    # The ids of the submissions with their JSON written, oldest first
    ids = []
    for key in store.keys(PREFIX):
        parts = key[len(PREFIX):].split("/")
        if len(parts) == 2 and parts[1] == JSON_NAME and is_id(parts[0]):
            ids.append(parts[0])
    return sorted(ids)


def write(path, data):
    temporary = f"{path}.part"
    with open(temporary, "wb") as file:
        file.write(data)
    os.replace(temporary, path)


def fetch(store, out_folder=OUT_FOLDER):
    '''
    Downloads every whole submission not yet in out_folder, and returns
    the ids of those it downloaded.
    '''
    fetched = []
    for submission_id in complete_ids(store):
        folder = os.path.join(out_folder, submission_id)
        if os.path.exists(os.path.join(folder, JSON_NAME)):
            continue
        os.makedirs(folder, exist_ok=True)
        data = store.get(f"{PREFIX}{submission_id}/{JSON_NAME}")
        if json.loads(data).get("image"):
            write(os.path.join(folder, PAGE_NAME),
                  store.get(f"{PREFIX}{submission_id}/{PAGE_NAME}"))
        # Last, as on the server, so a folder with its JSON is whole
        write(os.path.join(folder, JSON_NAME), data)
        fetched.append(submission_id)
    return fetched


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folder", help="a local store to read instead "
                        "of R2 (the server's SUBMISSIONS_FOLDER)")
    parser.add_argument("--out", default=OUT_FOLDER,
                        help="where to put them (default: %(default)s)")
    args = parser.parse_args()
    if args.folder:
        store = FolderStore(args.folder)
    else:
        environ = {key: value for key, value in os.environ.items()
                   if key != "SUBMISSIONS_FOLDER"}
        store = store_from_env(environ)
        if store is None:
            parser.error("Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
                         "R2_SECRET_ACCESS_KEY and R2_BUCKET, or use "
                         "--folder.")
    fetched = fetch(store, args.out)
    print(f"Fetched {len(fetched)} new submission"
          f"{'' if len(fetched) == 1 else 's'} into {args.out}.")
    for submission_id in fetched:
        print(f"  {submission_id}")


if __name__ == "__main__":
    main()
