# Calendar Integration Setup (5-10 min, no approval wait)

1. https://console.cloud.google.com/ -> New Project -> name it `aiaa-dev`
2. Left menu -> APIs & Services -> Library -> search "Google Calendar API" -> Enable
3. APIs & Services -> OAuth consent screen -> External -> fill app name + your email
   -> Add your own email under "Test users" -> Save
4. APIs & Services -> Credentials -> Create Credentials -> OAuth client ID
   -> Application type: **Desktop app** -> Create
5. Download JSON -> rename to `credentials.json` -> put it in this `calendar_integration/` folder
6. Install deps:
   ```
   pip install google-auth-oauthlib google-api-python-client
   ```
7. Run:
   ```
   python calendar_agent.py
   ```
   A browser window opens once -> log in with your Google account -> approve.
   `token.json` gets cached -> won't ask again.
8. Check your actual Google Calendar -> you'll see a real event titled
   "AIAA real-execution test" -> that's your proof this is real execution,
   not input validation.

If step 7 errors with "access blocked: app not verified" -> click
"Advanced" -> "Go to aiaa-dev (unsafe)" -> this is normal for apps in
testing mode with yourself as a test user, completely safe.
