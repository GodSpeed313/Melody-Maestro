import csv
import datetime
import os
import requests

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = os.environ.get("GITHUB_REPO", "GodSpeed313/Melody-Maestro")
LOG_FILE = "governance_log.csv"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def get_recent_commits(since_iso):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits"
    params = {"since": since_iso, "per_page": 100}
    response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def get_commit_files(sha):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{sha}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return [f["filename"] for f in response.json().get("files", [])]


def send_discord(status, message):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {"content": f"**[{status}]** {timestamp}: {message}"}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord failed: {e}")


def log_result(status, message):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Status", "Message"])
        writer.writerow([timestamp, status, message])


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    since = (now - datetime.timedelta(hours=24)).isoformat()

    commits = get_recent_commits(since)

    if not commits:
        status = "I'M ALIVE"
        message = f"No commits in the last 24 hours. {GITHUB_REPO} is stable."
        send_discord(status, message)
        log_result(status, message)
        print(f"[{status}] {message}")
        return

    py_changed = False
    readme_changed = False

    for commit in commits:
        files = get_commit_files(commit["sha"])
        for f in files:
            if f.endswith(".py"):
                py_changed = True
            if f.lower() == "readme.md":
                readme_changed = True

    if py_changed and not readme_changed:
        status = "I FAILED"
        message = f".py files changed in the last 24 hours without a README.md update. Governance drift detected in {GITHUB_REPO}."
    elif py_changed and readme_changed:
        status = "I'M ALIVE"
        message = f".py files changed and README.md updated in the same window. Coherent commit pattern in {GITHUB_REPO}."
    else:
        status = "I'M ALIVE"
        message = f"No .py changes in the last 24 hours. {GITHUB_REPO} is stable."

    send_discord(status, message)
    log_result(status, message)
    print(f"[{status}] {message}")


if __name__ == "__main__":
    main()
