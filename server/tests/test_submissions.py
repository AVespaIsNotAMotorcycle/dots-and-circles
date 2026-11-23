import io
import json

import pytest

from ocr.fetch_submissions import complete_ids, fetch
from server.submissions import (FolderStore, R2Store, RateLimiter, is_id,
                                new_id, store_from_env)


def test_new_id():
    first, second = new_id(), new_id()
    assert first != second and is_id(first)
    assert not is_id("../etc") and not is_id("") and not is_id("a/b")


def test_folder_store(tmp_path):
    store = FolderStore(str(tmp_path))
    assert store.keys() == []
    store.put("submissions/b/x.json", b"{}", "application/json")
    store.put("submissions/a/page.png", b"png", "image/png")
    # Half-written files aren't listed
    (tmp_path / "submissions" / "a" / "y.part").write_bytes(b"")
    assert store.keys() == ["submissions/a/page.png", "submissions/b/x.json"]
    assert store.get("submissions/a/page.png") == b"png"


class FakeS3:
    # Just enough of boto3's S3 client for R2Store
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(key for bucket, key in objects
                              if bucket == Bucket and key.startswith(Prefix))
                # Two pages, to check they're joined
                yield {"Contents": [{"Key": key} for key in keys[:1]]}
                yield ({"Contents": [{"Key": key} for key in keys[1:]]}
                       if keys[1:] else {})
        return Paginator()


def test_r2_store():
    client = FakeS3()
    store = R2Store("account", "key", "secret", "bucket", client=client)
    store.put("submissions/a/submission.json", b"{}", "application/json")
    store.put("submissions/b/submission.json", b"[]", "application/json")
    store.put("other/c", b"", "text/plain")
    assert store.keys() == ["submissions/a/submission.json",
                            "submissions/b/submission.json"]
    assert store.get("submissions/b/submission.json") == b"[]"


def test_store_from_env(tmp_path):
    assert store_from_env({}) is None
    store = store_from_env({"SUBMISSIONS_FOLDER": str(tmp_path)})
    assert isinstance(store, FolderStore)
    with pytest.raises(RuntimeError, match="R2_BUCKET"):
        store_from_env({"R2_ACCOUNT_ID": "a", "R2_ACCESS_KEY_ID": "b",
                        "R2_SECRET_ACCESS_KEY": "c"})


def test_rate_limiter():
    now = [0.0]
    limiter = RateLimiter(2, 100, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 99
    assert not limiter.allow("a")
    # Once the period has passed, the first ones are forgotten
    now[0] = 100.5
    assert limiter.allow("a")
    now[0] = 1000
    limiter.allow("c")
    assert set(limiter.times) == {"c"}


def test_fetch(tmp_path):
    store = FolderStore(str(tmp_path / "store"))
    store.put("submissions/2026-11-23T120000Z-aaaa/page.png", b"png",
              "image/png")
    store.put("submissions/2026-11-23T120000Z-aaaa/submission.json",
              json.dumps({"image": "page.png"}).encode(), "application/json")
    # A real page's, with no image
    store.put("submissions/2026-11-23T130000Z-bbbb/submission.json",
              json.dumps({"image": None}).encode(), "application/json")
    # Not yet whole: no JSON
    store.put("submissions/2026-11-23T140000Z-cccc/page.png", b"png",
              "image/png")
    assert complete_ids(store) == ["2026-11-23T120000Z-aaaa",
                                   "2026-11-23T130000Z-bbbb"]
    out = tmp_path / "out"
    assert fetch(store, str(out)) == complete_ids(store)
    assert (out / "2026-11-23T120000Z-aaaa" / "page.png").read_bytes() \
        == b"png"
    assert not (out / "2026-11-23T130000Z-bbbb" / "page.png").exists()
    # Fetched again, there's nothing new
    assert fetch(store, str(out)) == []
