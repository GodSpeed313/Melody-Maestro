import csv
import datetime
import json
import os
import sys
import requests

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GH_PAT = os.environ.get("GH_PAT", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "GodSpeed313/Melody-Maestro")
LOG_FILE = "governance_log.csv"

_ROOT = os.path.dirname(os.path.abspath(__file__))
IR_PATH = os.path.join(_ROOT, "governance", "ir.json")
TRACES_DIR = os.path.join(_ROOT, "governance", "traces")

HEADERS = {"Accept": "application/vnd.github+json"}
if GH_PAT:
    HEADERS["Authorization"] = f"Bearer {GH_PAT}"


# ── GitHub API helpers ────────────────────────────────────────────────────────

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


def migration_exists_in_repo():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/prisma/migrations"
    response = requests.get(url, headers=HEADERS, timeout=15)
    if response.status_code == 200:
        contents = response.json()
        return isinstance(contents, list) and len(contents) > 0
    return False


# ── Entity state builder ──────────────────────────────────────────────────────

def build_entity_state(commits, now):
    py_files_changed = False
    readme_changed = False
    schema_changed = False
    ir_changed_raw = False
    watcher_changed = False

    for commit in commits:
        for f in get_commit_files(commit["sha"]):
            if f.endswith(".py"):
                py_files_changed = True
            if f.lower() == "readme.md":
                readme_changed = True
            if f.endswith(".prisma"):
                schema_changed = True
            if f == "governance/ir.json":
                ir_changed_raw = True
            if f == "governance_watcher.py":
                watcher_changed = True

    # ReadmeCoherence fires when py files changed WITHOUT a readme update.
    # Encode as 1 so the `>= 1` condition in the resolver triggers correctly.
    py_changed = 1 if (py_files_changed and not readme_changed) else 0

    # IRSync fires when ir.json changed WITHOUT a matching watcher update.
    ir_changed = ir_changed_raw and not watcher_changed

    return {
        "py_changed":       py_changed,
        "readme_changed":   readme_changed,
        "schema_changed":   schema_changed,
        "migration_exists": migration_exists_in_repo(),
        "ir_changed":       ir_changed,
        "watcher_changed":  watcher_changed,
        "commit_count":     len(commits),
        "session_id":       f"governance-{now.strftime('%Y-%m-%d')}",
    }


# ── Trace / Discord / CSV ─────────────────────────────────────────────────────

def save_trace(trace, now):
    os.makedirs(TRACES_DIR, exist_ok=True)
    filename = f"trace-{now.strftime('%Y%m%d-%H%M%S')}.json"
    path = os.path.join(TRACES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return path


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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from pi_script.resolver import resolve

    now = datetime.datetime.now(datetime.timezone.utc)
    since = (now - datetime.timedelta(hours=24)).isoformat()

    with open(IR_PATH, encoding="utf-8") as f:
        ir = json.load(f)

    commits = get_recent_commits(since)

    if not commits:
        entity_state = {
            "py_changed":       0,
            "readme_changed":   False,
            "schema_changed":   False,
            "migration_exists": migration_exists_in_repo(),
            "ir_changed":       False,
            "watcher_changed":  False,
            "commit_count":     0,
            "session_id":       f"governance-{now.strftime('%Y-%m-%d')}",
        }
        trigger_type = "heartbeat"
    else:
        entity_state = build_entity_state(commits, now)
        trigger_type = "event"

    state = {
        "trigger_type": trigger_type,
        "entity":       "MelodyMaestroRepo",
        "entity_state": entity_state,
    }

    trace, rendered, exit_code = resolve(ir, state)
    print(rendered)

    trace_path = save_trace(trace, now)
    print(f"Trace saved: {trace_path}")

    violations = [c for c in trace.get("constraints", []) if c["status"] == "violated"]
    system_state = trace.get("system_state", "running")

    if violations:
        names = ", ".join(v["name"] for v in violations)
        status = "I FAILED"
        message = (
            f"Policy violations in {GITHUB_REPO}: {names}. "
            f"System state: {system_state}. "
            f"Final action: {trace.get('final_action', 'unknown')}."
        )
    else:
        status = "I'M ALIVE"
        message = f"All constraints satisfied in {GITHUB_REPO}. System state: {system_state}."

    send_discord(status, message)
    log_result(status, message)
    print(f"[{status}] {message}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
