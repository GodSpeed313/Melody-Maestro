# Renewing the `PISCRIPTGOVERNANCE` PAT

The Governance Watcher workflow (`.github/workflows/governance.yml`) authenticates with a
**fine-grained Personal Access Token** stored as the repo secret `PISCRIPTGOVERNANCE`. It is used
to check out this repo **and** `GodSpeed313/Continuum` (the Pi Script engine) and to push governance
trace commits back to `governance/traces/`.

Fine-grained PATs default to a **60-day** lifetime. When the token expires, `actions/checkout` fails
authentication, the whole run dies before the watcher can post, and the Discord `#pi-logs` channel
just goes quiet — exactly what happened for ~10 days in June 2026 before it was noticed.

## The warning mechanism (so it is never silent again)

The workflow's first step (`Check PISCRIPTGOVERNANCE PAT expiry`) reads the token's real expiry from
GitHub's `github-authentication-token-expiration` response header on every run (every 12h) and posts
to Discord:

| Condition | Discord message |
|---|---|
| > 14 days left | (silent — logged to the Actions run only) |
| ≤ 14 days left | ⚠️ heads-up, plan to regenerate |
| ≤ 3 days left | 🔴 URGENT, regenerate now |
| already expired / revoked (HTTP 401/403) | 🔴 CRITICAL, watcher is DOWN; the run also fails loudly |

A final `if: failure()` step nets **any other** run failure so the channel is never silently quiet.

The expiry is read live from the header, so **there is no date hardcoded anywhere** that has to be
updated when you rotate the token.

## How to regenerate

1. GitHub → **Settings → Developer settings → Fine-grained tokens → Generate new token**.
2. **Resource owner:** `GodSpeed313`.
3. **Repository access:** only these two repos — `GodSpeed313/Melody-Maestro` and
   `GodSpeed313/Continuum`.
4. **Repository permissions: Contents → Read and write.**
   - A fine-grained token applies ONE permission set to every selected repo — you cannot give
     Melody-Maestro write and Continuum read-only in a single token. Since the workflow must push
     trace commits to Melody-Maestro, set **Contents → Read and write**; it covers both repos.
     Continuum only needs read, so its write grant is unused (harmless) collateral.
   - Leave every other permission at "No access". (Metadata read-only is added automatically.)
5. **Expiration:** pick the lifetime you want. A longer expiry means fewer rotations; a shorter one
   is safer. Either way the ⚠️/🔴 warnings above give ~2 weeks of lead time before it lapses.
6. Copy the new token (`github_pat_…`).
7. This repo → **Settings → Secrets and variables → Actions → `PISCRIPTGOVERNANCE` → Update secret**,
   paste the new value.
8. Verify: **Actions → Governance Watcher → Run workflow** (manual `workflow_dispatch`). The
   `Check PISCRIPTGOVERNANCE PAT expiry` step log should print `PAT healthy: N days remaining`, and
   the run should complete green with a fresh trace committed.

## Notes

- The Discord webhook is a separate secret, `PILogs` — it does not expire and is unaffected by PAT
  rotation.
- If you switch to a non-expiring token, the check authenticates but finds no expiry header and posts
  a one-line ⚠️ note confirming that; it will not nag afterward.
