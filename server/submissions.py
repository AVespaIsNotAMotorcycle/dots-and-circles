'''
Where POST /submissions keeps what users send: their page, its boxes and
their corrections, for improving how the page reader draws boxes. They
aren't public, and nothing trains on them until they've been looked over.

Each submission is two objects under submissions/<id>/: page.png (left
out for a real page, which the server already has) and submission.json,
written last, so a submission without one is incomplete and is skipped.

Two stores keep them, with the same methods:

- R2Store: a Cloudflare R2 bucket, used on Heroku, whose dyno's own disk
  is wiped on every restart. It needs boto3.
- FolderStore: a folder on disk, for running the server locally and for
  the tests.

store_from_env picks one from the environment (see server/README.md).
`python -m ocr.fetch_submissions` downloads them.
'''

import datetime
import os
import threading
import time
import uuid

PREFIX = "submissions/"
PAGE_NAME = "page.png"
JSON_NAME = "submission.json"


def new_id(now=None):
    # A submission's id: when it came (UTC), which sorts them, and a random
    # part, so two in the same second don't collide
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def is_id(name):
    # Whether name could be a submission id, and so is safe as a folder name
    return (bool(name) and len(name) <= 64 and name[0].isdigit()
            and all(c.isalnum() or c in "-" for c in name))


class FolderStore:
    def __init__(self, folder):
        self.folder = folder

    def put(self, key, data, content_type):
        path = os.path.join(self.folder, *key.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written whole, then moved into place, so it's never seen half
        # written
        temporary = f"{path}.part"
        with open(temporary, "wb") as file:
            file.write(data)
        os.replace(temporary, path)

    def get(self, key):
        with open(os.path.join(self.folder, *key.split("/")), "rb") as file:
            return file.read()

    def keys(self, prefix=PREFIX):
        # Every key under prefix, sorted
        root = os.path.join(self.folder, *prefix.rstrip("/").split("/"))
        found = []
        for folder, _, files in os.walk(root):
            for name in files:
                if name.endswith(".part"):
                    continue
                path = os.path.relpath(os.path.join(folder, name),
                                       self.folder)
                found.append(path.replace(os.sep, "/"))
        return sorted(found)


class R2Store:
    '''
    A Cloudflare R2 bucket, through its S3-compatible API. The token needs
    Object Read & Write on the bucket for the server, and only Object Read
    for fetching.
    '''

    def __init__(self, account_id, access_key_id, secret_access_key, bucket,
                 client=None):
        if client is None:
            # Imported here, so nothing else needs boto3
            import boto3
            client = boto3.client(
                "s3",
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name="auto")
        self.client = client
        self.bucket = bucket

    def put(self, key, data, content_type):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType=content_type)

    def get(self, key):
        return self.client.get_object(Bucket=self.bucket,
                                      Key=key)["Body"].read()

    def keys(self, prefix=PREFIX):
        found = []
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            found.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(found)


R2_VARIABLES = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                "R2_BUCKET")


def store_from_env(environ=os.environ):
    '''
    The store the environment names: R2 if all of R2_VARIABLES are set, a
    folder if SUBMISSIONS_FOLDER is, and otherwise None (the server then
    takes no submissions). Some but not all of the R2 ones is a mistake,
    so it raises rather than quietly taking none.
    '''
    present = [name for name in R2_VARIABLES if environ.get(name)]
    if present:
        missing = [name for name in R2_VARIABLES if not environ.get(name)]
        if missing:
            raise RuntimeError(f"Set {', '.join(missing)} too, or none of "
                               f"{', '.join(R2_VARIABLES)}.")
        return R2Store(*(environ[name] for name in R2_VARIABLES))
    if environ.get("SUBMISSIONS_FOLDER"):
        return FolderStore(environ["SUBMISSIONS_FOLDER"])
    return None


class RateLimiter:
    '''
    At most limit submissions per address in any period seconds. It's kept
    in memory only, so each gunicorn worker has its own and a restart
    forgets it: it's there to stop a script filling the bucket, not to
    count exactly.
    '''

    def __init__(self, limit, period, clock=time.monotonic):
        self.limit = limit
        self.period = period
        self.clock = clock
        self.times = {}
        self.lock = threading.Lock()

    def allow(self, address):
        # Records a submission from address and returns True, or returns
        # False if it's had its limit
        now = self.clock()
        with self.lock:
            # Forget everyone's old times, so the dict doesn't grow forever
            for key in list(self.times):
                kept = [t for t in self.times[key] if now - t < self.period]
                if kept:
                    self.times[key] = kept
                else:
                    del self.times[key]
            recent = self.times.setdefault(address, [])
            if len(recent) >= self.limit:
                return False
            recent.append(now)
            return True
