import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from youtube_sentiment.youtube_api import fetch_comments, load_cached_comments, parse_video_id, save_cached_comments


class FakeResponse:
    ok = True

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(next(self.payloads))


def item(text):
    return {"snippet": {"topLevelComment": {"snippet": {"textOriginal": text}}}}


class YouTubeAPITests(unittest.TestCase):
    def test_parse_video_id(self):
        values = (
            "dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=3",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(parse_video_id(value), "dQw4w9WgXcQ")

    def test_parse_video_id_rejects_untrusted_path_input(self):
        with self.assertRaises(ValueError):
            parse_video_id("../../secret")

    def test_fetch_comments_paginates_and_respects_limit(self):
        session = FakeSession(
            [
                {"items": [item("one"), item("two")], "nextPageToken": "next"},
                {"items": [item("three"), item("four")]},
            ]
        )
        comments = fetch_comments("key", "dQw4w9WgXcQ", max_comments=3, session=session)
        self.assertEqual(comments, ["one", "two", "three"])
        self.assertEqual(session.calls[1][1]["pageToken"], "next")

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comments.json"
            save_cached_comments(path, "dQw4w9WgXcQ", ["hello"])
            self.assertEqual(load_cached_comments(path, "dQw4w9WgXcQ", timedelta(hours=1)), ["hello"])
            payload = json.loads(path.read_text())
            self.assertEqual(payload["video_id"], "dQw4w9WgXcQ")


if __name__ == "__main__":
    unittest.main()
