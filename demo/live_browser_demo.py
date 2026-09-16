"""
Real-time, real-browser AIAA demo.

This is the "for the video" script:
1. Sends the task to AIAA's real API (gets the real risk-based decision:
   intervention required or not, autonomy level, etc.)
2. If intervention is required, it PAUSES and prints a human-approval
   prompt (press Enter to approve) — this is real, not simulated.
3. Then drives an ACTUAL browser (Playwright/Chromium) to open the mock
   bill payment form, fill it with the task's real data, and click submit
   — a real DOM form, real fields, real click, real confirmation message.
4. Times every phase separately so you can show "decision: 0.3s,
   browser automation: 2.1s" on camera.

Setup (one-time):
    pip install playwright --break-system-packages
    playwright install chromium

Run:
    python demo/live_browser_demo.py
(Server must be running: python run_server.py)
"""

import time
import os
import requests
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8001"  # change to your server's port
FORM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_portal", "bill_payment.html"))
FORM_URL = f"file:///{FORM_PATH.replace(os.sep, '/')}"


def get_aiaa_decision(amount, payee, due):
    """Step 1: real API call to AIAA — this is where the risk/intervention decision happens."""
    print(f"\n[1/3] Sending task to AIAA: amount=₹{amount}, payee={payee}")
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/tasks",
        json={
            "task_type": "bill_payment",
            "user_id": "live_demo_user",
            "input_data": {"amount": amount, "payee": payee, "due": due},
        },
        timeout=15,
    )
    decision_time = time.time() - t0
    result = resp.json()
    print(f"      -> Decision made in {decision_time:.2f}s")
    print(f"      -> Status: {result['status']}")
    print(f"      -> Intervention required: {result['intervention_required']}")
    return result, decision_time


def human_approval_gate(intervention_required):
    """Step 2: if AIAA says this needs a human check, actually wait for one."""
    if not intervention_required:
        print("[2/3] No human approval needed — proceeding autonomously.")
        return 0.0
    print("\n[2/3] ⚠️  AIAA flagged this as high-risk. Human approval required.")
    t0 = time.time()
    input("      Press Enter to approve this payment...")
    return time.time() - t0


def run_browser_automation(amount, payee, due):
    """Step 3: real browser — actually opens the form, fills it, submits it."""
    print("\n[3/3] Launching real browser to complete the payment...")
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=400)  # slow_mo so it's watchable on camera
        page = browser.new_page()
        page.goto(FORM_URL)
        page.fill("#amount", str(amount))
        page.fill("#payee", payee)
        page.fill("#due", due)
        page.click("#submit-btn")
        page.wait_for_selector("#confirmation", state="visible", timeout=5000)
        conf_text = page.inner_text("#confirmation")
        time.sleep(1.5)  # let the confirmation stay on screen for the camera
        browser.close()
    automation_time = time.time() - t0
    print(f"      -> {conf_text.strip()}")
    print(f"      -> Browser automation took {automation_time:.2f}s")
    return automation_time


def main():
    print("=" * 55)
    print("AIAA LIVE DEMO — Real Decision + Real Browser Automation")
    print("=" * 55)

    amount = float(input("Amount (₹): ").strip())
    payee = input("Payee: ").strip()
    due = input("Due date (YYYY-MM-DD): ").strip()

    result, decision_time = get_aiaa_decision(amount, payee, due)
    approval_time = human_approval_gate(result["intervention_required"])
    automation_time = run_browser_automation(amount, payee, due)

    total = decision_time + approval_time + automation_time
    print("\n" + "=" * 55)
    print("TIMING BREAKDOWN")
    print("=" * 55)
    print(f"  AIAA decision (risk + intervention check): {decision_time:.2f}s")
    print(f"  Human approval wait:                       {approval_time:.2f}s")
    print(f"  Browser automation (fill + submit):        {automation_time:.2f}s")
    print(f"  TOTAL:                                     {total:.2f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()