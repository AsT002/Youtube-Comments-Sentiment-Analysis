"""A small, testable client for retrieving top-level YouTube comments."""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


class YouTubeAPIError(RuntimeError):
    """An actionable YouTube API failure."""


def parse_video_id(value: str) -> str:
    candidate = value.strip()
    if VIDEO_ID_RE.fullmatch(candidate):
        return candidate
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    hostname = (parsed.hostname or "").lower()
    if hostname in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            candidate = parsed.path.strip("/").split("/")[1]
    if not VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError("Expected an 11-character YouTube video ID or a supported YouTube URL.")
    return candidate


def build_session():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise RuntimeError("Install requests before accessing the YouTube API.") from exc
    retries = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _error_message(response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {})
        reasons = [item.get("reason") for item in error.get("errors", []) if item.get("reason")]
        message = error.get("message") or response.text
        return f"{message} ({', '.join(reasons)})" if reasons else str(message)
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"


def fetch_comments(api_key: str, video_id: str, max_comments: Optional[int] = 1_000, timeout: float = 15.0, session=None) -> List[str]:
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not configured.")
    video_id = parse_video_id(video_id)
    client = session or build_session()
    comments: List[str] = []
    page_token: Optional[str] = None
    while True:
        remaining = None if max_comments is None else max_comments - len(comments)
        if remaining is not None and remaining <= 0:
            break
        params = {"part": "snippet", "videoId": video_id, "key": api_key, "textFormat": "plainText", "maxResults": min(100, remaining) if remaining is not None else 100}
        if page_token:
            params["pageToken"] = page_token
        response = client.get(COMMENTS_URL, params=params, timeout=timeout)
        if not response.ok:
            raise YouTubeAPIError(_error_message(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise YouTubeAPIError("YouTube returned an invalid JSON response.") from exc
        for item in payload.get("items", []):
            try:
                text = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            except (KeyError, TypeError):
                continue
            if text:
                comments.append(text)
            if max_comments is not None and len(comments) >= max_comments:
                break
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return comments


def load_cached_comments(path: Path, video_id: str, max_age: timedelta) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("video_id") != video_id:
            return None
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now(timezone.utc) - fetched_at > max_age:
            return None
        comments = payload.get("comments")
        return comments if isinstance(comments, list) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached_comments(path: Path, video_id: str, comments: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_id": video_id, "fetched_at": datetime.now(timezone.utc).isoformat(), "comments": comments}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
