"""
Real Google Calendar execution module — replaces "input validation = success"
with actual create -> verify round trip against your real Google account.

Setup (5 min, no approval wait):
1. console.cloud.google.com -> New Project -> "aiaa-dev"
2. APIs & Services -> Library -> "Google Calendar API" -> Enable
3. APIs & Services -> Credentials -> Create Credentials -> OAuth client ID
   -> Desktop app -> Create -> Download JSON -> save as credentials.json here
4. pip install google-auth-oauthlib google-api-python-client
First run opens browser login once; token.json caches after.
"""
import os, datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CREDS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")


def _get_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def create_event(summary, start_iso, end_iso, description="", calendar_id="primary"):
    service = _get_service()
    body = {"summary": summary, "description": description,
            "start": {"dateTime": start_iso, "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end_iso, "timeZone": "Asia/Kolkata"}}
    return service.events().insert(calendarId=calendar_id, body=body).execute()


def verify_event(event_id, calendar_id="primary"):
    service = _get_service()
    try:
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        return event.get("status") in ("confirmed", "tentative")
    except Exception:
        return False


def run_calendar_task(summary, start_iso, end_iso, description=""):
    import time, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "logging"))
    from db import log_task_event, update_outcome  # noqa

    task_id = log_task_event(task_type="calendar_event",
                              context={"summary": summary, "start": start_iso, "end": end_iso})
    try:
        created = create_event(summary, start_iso, end_iso, description)
        success = verify_event(created["id"])
        update_outcome(task_id, outcome_success=success,
                        outcome_detail={"event_id": created["id"], "html_link": created.get("htmlLink")})
        return {"task_id": task_id, "success": success, "event": created}
    except Exception as e:
        update_outcome(task_id, outcome_success=False, outcome_detail={"error": str(e)})
        return {"task_id": task_id, "success": False, "error": str(e)}


if __name__ == "__main__":
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    end = now + datetime.timedelta(minutes=30)
    print(run_calendar_task("AIAA real-execution test", now.isoformat()+"Z", end.isoformat()+"Z",
                             "Created by AIAA to prove real execution, not mock validation."))