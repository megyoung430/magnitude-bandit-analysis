"""A minimal Notion API client: the few calls the code-map sync needs (the network edge).

It uses only the standard library. The HTTP call itself is injectable (``send``), so tests run
against a fake without a network or a token.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator

API = "https://api.notion.com/v1"
# The version whose endpoints address a single-source database by its database id.
NOTION_VERSION = "2022-06-28"
MAX_RETRIES = 5
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# send(method, url, headers, body bytes or None) -> (status, response headers, body bytes)
Send = Callable[[str, str, dict, bytes | None], tuple[int, dict, bytes]]


class NotionError(RuntimeError):
    """A Notion request that failed, with the API's own message."""


def urllib_send(
    method: str, url: str, headers: dict, body: bytes | None
) -> tuple[int, dict, bytes]:
    """One HTTP request with urllib; HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def retry_delay(headers: dict, attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After, else exponential backoff."""
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    return float(value) if value else min(2.0**attempt, 30.0)


class NotionClient:
    """Notion calls made with one integration token.

    Args:
        token: The integration token (secret; never logged).
        send: The HTTP call; defaults to urllib.
        sleep: Called with seconds to wait between retries; injectable for tests.
    """

    def __init__(self, token: str, send: Send = urllib_send, sleep: Callable = time.sleep):
        if not token:
            raise NotionError("no Notion token: set NOTION_TOKEN in .env (see .env.example)")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        self._send, self._sleep = send, sleep

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        """One API call, retried on rate limits and server errors.

        Raises:
            NotionError: For any other error status, or when retries run out; the message
                carries the API's code and message (e.g. an unshared page is "object_not_found").
        """
        data = json.dumps(body).encode("utf-8") if body is not None else None
        for attempt in range(MAX_RETRIES + 1):
            status, headers, raw = self._send(method, f"{API}/{path}", self._headers, data)
            if status < 300:
                return json.loads(raw or b"{}")
            if status not in RETRY_STATUSES or attempt == MAX_RETRIES:
                break
            self._sleep(retry_delay(headers, attempt))
        try:
            detail = json.loads(raw)
            message = f"{detail.get('code')}: {detail.get('message')}"
        except (ValueError, AttributeError):
            message = raw[:300].decode("utf-8", "replace")
        raise NotionError(f"{method} {path} failed with {status}: {message}")

    def query_database(self, database_id: str) -> Iterator[dict]:
        """Every row (page object) of a database, following pagination."""
        body: dict = {"page_size": 100}
        while True:
            page = self.request("POST", f"databases/{database_id}/query", body)
            yield from page["results"]
            if not page.get("has_more"):
                return
            body["start_cursor"] = page["next_cursor"]

    def create_row(self, database_id: str, properties: dict, children: list[dict]) -> str:
        """Create a row with its first batch of page content; returns the new page id."""
        body = {"parent": {"database_id": database_id}, "properties": properties}
        if children:
            body["children"] = children
        return self.request("POST", "pages", body)["id"]

    def append_blocks(self, page_id: str, children: list[dict]) -> None:
        """Append blocks to the end of a page."""
        self.request("PATCH", f"blocks/{page_id}/children", {"children": children})

    def update_properties(self, page_id: str, properties: dict) -> None:
        """Set some properties of a row."""
        self.request("PATCH", f"pages/{page_id}", {"properties": properties})

    def archive_page(self, page_id: str) -> None:
        """Move a page to the trash (restorable from Notion for 30 days)."""
        self.request("PATCH", f"pages/{page_id}", {"archived": True})
