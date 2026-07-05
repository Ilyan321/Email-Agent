#!/usr/bin/env python3
"""
triage_emails.py
=================
Production-grade Gmail inbox triaging bot.

Pipeline:
  1. Load credentials (local .env file OR exported environment / GitHub Actions secrets).
  2. Authenticate to Gmail via OAuth2 refresh-token rotation (no IMAP/SMTP).
  3. Pull unread messages from the primary inbox.
  4. Extract clean plaintext bodies (HTML/CSS stripped).
  5. Send each email to Groq (llama-3.3-70b-versatile) for structured JSON triage.
  6. If reply-worthy, create a threaded Gmail DRAFT reply (never auto-sent).
  7. Post a draft-aware summary card to Slack.
  8. Mark the processed message as read.

Works identically in two modes:
  - Local Mode: `python triage_emails.py` inside a venv, with a `.env` file present.
  - Cloud Mode: unattended execution inside a GitHub Actions runner, values injected
    via encrypted Repository Secrets as real environment variables.

Author: Automation Engineering
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from email.mime.text import MIMEText
from html import unescape
from typing import Any, Optional

# ----------------------------------------------------------------------------
# 0. ENVIRONMENT BOOTSTRAP
# ----------------------------------------------------------------------------
# Load a local .env file if present. In GitHub Actions, no .env file exists, so
# this is a harmless no-op and the real environment variables (populated from
# encrypted secrets) are used instead. This one call is what lets the exact
# same script run unmodified in both Local Mode and Cloud Mode.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from googleapiclient.discovery import build  # noqa: E402
from googleapiclient.errors import HttpError  # noqa: E402
from google.oauth2.credentials import Credentials  # noqa: E402
from google.auth.transport.requests import Request  # noqa: E402
from groq import Groq  # noqa: E402
import requests  # noqa: E402

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("triage_emails")

# ----------------------------------------------------------------------------
# CONSTANTS / CONFIG
# ----------------------------------------------------------------------------
REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "SLACK_BOT_TOKEN",
    "GMAIL_CLIENT_ID",
    "GMAIL_CLIENT_SECRET",
    "GMAIL_REFRESH_TOKEN",
]

# Optional, sensible defaults so the required-vars list stays exactly at 5.
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "#inbox-triage")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
GMAIL_TOKEN_URI = "https://oauth2.googleapis.com/token"

TRIAGE_SYSTEM_PROMPT = """You are an executive email-triage assistant.

You will be given the Subject and Body of a single email. Evaluate it and
respond with STRICT JSON ONLY (no markdown fences, no commentary, no extra
keys) matching exactly this schema:

{
  "is_reply_worthy": true or false,
  "summary": "A detailed 2 to 3 sentence breakdown explaining who the sender is, the explicit purpose or context of the message, and any direct action items, questions, or deadlines mentioned.",
  "professional_reply": "A polished, professionally written email response
    contextually tailored to the incoming message if is_reply_worthy is
    true. If is_reply_worthy is false, this field MUST be an empty string."
}

Rules:
- "is_reply_worthy" is true only for emails that genuinely require a human
  response (direct questions, requests, meeting proposals, negotiations,
  client/customer correspondence). Mark newsletters, receipts, automated
  notifications, marketing blasts, and pure FYI threads as false.
- "summary" MUST be a comprehensive paragraph spanning exactly 2 to 3 complete sentences. Do not make it short or simple. Provide enough context so that the reader understands the entire situation clearly without opening the email.
- "professional_reply" must be ready to send as-is: appropriate greeting,
  body, and sign-off placeholder ("Best regards,"), written in a warm but
  professional tone. Do not invent facts not present in the original email.
- Output must be valid JSON and nothing else.
"""


# ----------------------------------------------------------------------------
# 1. ENVIRONMENT VALIDATION
# ----------------------------------------------------------------------------
def validate_environment() -> None:
    """Fail fast with a clear message if any required secret is missing."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        log.error(
            "Missing required environment variable(s): %s. "
            "Set them in a local .env file (Local Mode) or as GitHub "
            "Repository Secrets (Cloud Mode).",
            ", ".join(missing),
        )
        sys.exit(1)
    log.info("Environment validated: all %d required variables present.", len(REQUIRED_ENV_VARS))


# ----------------------------------------------------------------------------
# 2. GMAIL AUTH + INGESTION
# ----------------------------------------------------------------------------
def get_gmail_service():
    """Build an authenticated Gmail API service by rotating the refresh token."""
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GMAIL_REFRESH_TOKEN"],
        client_id=os.environ["GMAIL_CLIENT_ID"],
        client_secret=os.environ["GMAIL_CLIENT_SECRET"],
        token_uri=GMAIL_TOKEN_URI,
        scopes=GMAIL_SCOPES,
    )

    try:
        creds.refresh(Request())
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to refresh Gmail OAuth2 access token: %s", exc)
        sys.exit(1)

    log.info("Gmail OAuth2 access token refreshed successfully.")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_unread_messages(service) -> list[dict]:
    """Return raw message stubs ({'id', 'threadId'}) for unread inbox mail."""
    try:
        response = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread", labelIds=["INBOX"])
            .execute()
        )
    except HttpError as exc:
        log.error("Gmail API error while listing messages: %s", exc)
        sys.exit(1)

    return response.get("messages", [])


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def strip_html(raw_html: str) -> str:
    """Strip tags/CSS/scripts from an HTML email body, leaving clean text."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_part_data(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")


def extract_body(payload: dict) -> str:
    """
    Recursively walk the MIME payload tree, preferring text/plain and
    falling back to a stripped-down version of text/html.
    """
    plain_text: Optional[str] = None
    html_text: Optional[str] = None

    def walk(part: dict) -> None:
        nonlocal plain_text, html_text
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        if mime_type == "text/plain" and data and plain_text is None:
            plain_text = _decode_part_data(data)
        elif mime_type == "text/html" and data and html_text is None:
            html_text = _decode_part_data(data)

        for sub_part in part.get("parts", []) or []:
            walk(sub_part)

    walk(payload)

    if plain_text:
        return plain_text.strip()
    if html_text:
        return strip_html(html_text)
    return "(No readable body content found.)"


def get_message_details(service, msg_id: str) -> dict[str, Any]:
    """Fetch and normalize the fields we care about for a single message."""
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=msg_id, format="full")
        .execute()
    )

    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    return {
        "id": msg["id"],
        "thread_id": msg["threadId"],
        "message_id_header": _header(headers, "Message-ID"),
        "subject": _header(headers, "Subject") or "(No Subject)",
        "sender": _header(headers, "From") or "(Unknown Sender)",
        "body": extract_body(payload),
    }


def mark_as_read(service, msg_id: str) -> None:
    try:
        service.users().messages().modify(
            userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()
    except HttpError as exc:
        log.warning("Could not mark message %s as read: %s", msg_id, exc)


def create_draft_reply(
    service,
    to_address: str,
    subject: str,
    reply_body: str,
    thread_id: str,
    original_message_id_header: str,
) -> bool:
    """
    Create a threaded Gmail draft so a human can review and send it manually.

    IMPORTANT: This function NEVER sends mail. It exclusively uses
    `service.users().drafts().create()`, which saves the composed reply into
    the Gmail Drafts folder, correctly threaded via `threadId` plus the
    `In-Reply-To`/`References` headers. A human must open Gmail and hit
    "Send" themselves. Returns True on success, False if draft creation failed.
    """
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"

    message = MIMEText(reply_body)
    message["to"] = to_address
    message["subject"] = reply_subject
    if original_message_id_header:
        message["In-Reply-To"] = original_message_id_header
        message["References"] = original_message_id_header

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    try:
        service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw, "threadId": thread_id}},
        ).execute()
        log.info("Draft reply saved to Gmail Drafts folder (thread %s). Not sent.", thread_id)
        return True
    except HttpError as exc:
        log.error("Failed to create draft reply for thread %s: %s", thread_id, exc)
        return False


# ----------------------------------------------------------------------------
# 3. GROQ TRIAGE
# ----------------------------------------------------------------------------
def triage_with_groq(client: Groq, subject: str, body: str) -> dict[str, Any]:
    """Send subject/body to Groq and force a strict JSON triage response."""
    # Guard against pathologically large bodies blowing the context window.
    trimmed_body = body[:8000]

    user_prompt = f"Subject: {subject}\n\nBody:\n{trimmed_body}"

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw_content = completion.choices[0].message.content
        result = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        log.error("Groq returned invalid JSON: %s", exc)
        result = {
            "is_reply_worthy": False,
            "summary": "Automated triage failed to parse a valid response.",
            "professional_reply": "",
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Groq API call failed: %s", exc)
        result = {
            "is_reply_worthy": False,
            "summary": "Automated triage failed due to an API error.",
            "professional_reply": "",
        }

    # Defensive normalization in case the model omits a key.
    result.setdefault("is_reply_worthy", False)
    result.setdefault("summary", "")
    result.setdefault("professional_reply", "")
    return result


# ----------------------------------------------------------------------------
# 4. SLACK NOTIFICATION
# ----------------------------------------------------------------------------
def post_to_slack(
    subject: str, sender: str, triage: dict[str, Any], draft_created: bool
) -> None:
    """
    Post a formatted triage card to Slack via chat.postMessage.

    This workflow never auto-sends email — it only ever creates Gmail
    drafts for a human to review. The status badge reflects that:
      - "📝 Draft Created"      -> a threaded Gmail draft is waiting for review/send.
      - "✅ Reply Recommended"  -> the model flagged this as reply-worthy, but the
                                    draft could not be saved (see logs); no draft exists.
      - "⏭️ No Reply Needed"    -> triaged as informational, no action required.
    """
    if triage["is_reply_worthy"] and draft_created:
        status_badge = "📝 Draft Created"
    elif triage["is_reply_worthy"]:
        status_badge = "✅ Reply Recommended"
    else:
        status_badge = "⏭️ No Reply Needed"

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*New Email Triaged*\n"
                    f"*From:* {sender}\n"
                    f"*Subject:* {subject}\n"
                    f"*Status:* {status_badge}\n"
                    f"*Summary:* {triage['summary']}"
                ),
            },
        }
    ]

    if triage["is_reply_worthy"] and triage["professional_reply"]:
        if draft_created:
            reply_caption = "*Draft Reply (waiting in Gmail Drafts — review & send manually):*"
        else:
            reply_caption = "*Suggested Reply (draft creation failed — copy manually if needed):*"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{reply_caption}\n>{triage['professional_reply']}",
                },
            }
        )

    fallback_text = (
        f"Draft created for: {subject}"
        if draft_created
        else f"New email triaged: {subject}"
    )

    payload = {
        "channel": SLACK_CHANNEL,
        "text": fallback_text,  # fallback text for notifications
        "blocks": blocks,
    }
    headers = {
        "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
        "Content-Type": "application/json; charset=utf-8",
    }

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload,
            timeout=15,
        )
        data = resp.json()
        if not data.get("ok"):
            log.warning("Slack API responded with an error: %s", data.get("error"))
        else:
            log.info("Slack notification posted to %s.", SLACK_CHANNEL)
    except requests.RequestException as exc:
        log.warning("Slack notification failed (network error): %s", exc)


# ----------------------------------------------------------------------------
# 5. MAIN ORCHESTRATION
# ----------------------------------------------------------------------------
def main() -> None:
    log.info("Starting email triage run.")
    validate_environment()

    gmail_service = get_gmail_service()
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    stubs = list_unread_messages(gmail_service)

    if not stubs:
        log.info("Inbox zero: no unread messages found. Exiting cleanly.")
        sys.exit(0)

    log.info("Found %d unread message(s). Beginning triage.", len(stubs))

    processed, failed = 0, 0

    for stub in stubs:
        msg_id = stub["id"]
        try:
            details = get_message_details(gmail_service, msg_id)
            log.info("Triaging message %s | Subject: %s", msg_id, details["subject"])

            triage = triage_with_groq(groq_client, details["subject"], details["body"])

            # Create the Gmail draft FIRST (if warranted) so the Slack status
            # badge can accurately report whether a draft actually exists.
            draft_created = False
            if triage["is_reply_worthy"] and triage["professional_reply"]:
                draft_created = create_draft_reply(
                    gmail_service,
                    to_address=details["sender"],
                    subject=details["subject"],
                    reply_body=triage["professional_reply"],
                    thread_id=details["thread_id"],
                    original_message_id_header=details["message_id_header"],
                )

            post_to_slack(details["subject"], details["sender"], triage, draft_created)

            mark_as_read(gmail_service, msg_id)
            processed += 1

        except Exception as exc:  # noqa: BLE001
            # Never let one bad message kill the whole batch run.
            log.error("Failed to process message %s: %s", msg_id, exc, exc_info=True)
            failed += 1
            continue

    log.info("Triage run complete. Processed: %d | Failed: %d", processed, failed)

    if failed and not processed:
        # Signal a non-zero exit for CI visibility only if nothing succeeded.
        sys.exit(1)


if __name__ == "__main__":
    main()
