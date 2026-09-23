# GitHub PR reviewer

**Checks PRs in monitored GitHub repositories and generates reviews using codex.**

Python standard-library runner for this Raspberry Pi. Uses `gh` for GitHub operations and the existing Codex ChatGPT login for Astra reviews. Runs sequentially under a nonblocking file lock.

## Repositories and defaults

`repos.json` enables `discover_local_repositories`: only immediate child checkouts inside `monitored-repos/` (next to `repos.json`) with a `.git` directory/file and a GitHub `origin` are monitored. Adding a checkout there adds it on the next invocation; repositories outside that directory are excluded. SSH and HTTPS GitHub origins work; personal checkouts are only inspected for their origin, never updated. Runtime caches live separately under `var/`.

In discovery mode, explicit entries override discovered repositories; entries without a checkout in `monitored-repos/` are ignored:

```json
"repositories": [
  {"name": "owner/repo", "enabled": true, "model": "gpt-6-astra"},
  {"name": "owner/ignored-repo", "enabled": false}
]
```

Models: `gpt-6-astra` (default), `gpt-5.6-sol`. No model or API-billing fallback.

Set the global reasoning level in `repos.json`:

```json
"reasoning_effort": "high"
```

Allowed values: `low`, `medium`, `high` (default), `xhigh`, `max`, `ultra`, matching the installed Codex model catalog for Sol and Astra. The runner passes this as [`model_reasoning_effort`](https://learn.chatgpt.com/docs/config-file/config-reference). Changes apply to the next model invocation, including pending reviews; cached results are reused. No cron restart is needed. The selected level is recorded in the activity log.

- First activation baselines existing ready PR revisions without reviewing them. New heads and drafts becoming ready are eligible; bots and fork PRs are included.
- One summary issue comment per PR head SHA, under the authenticated GitHub account. No approvals, merges, or automatic code changes.
- Completed revisions stay completed when only the base moves. For unfinished reviews, base movement discards output and retries that head against the new base.
- Defaults: 3 reviews per invocation, 15 minutes per model call, 3 attempts per generation/publication stage. Increasing retry delays; GitHub rate-limit responses defer longer.
- Stale heads and closed PRs are discarded; drafts wait. A push immediately after the final check can still race publication, so every comment identifies its reviewed commit.

## Setup

Requires Linux with unprivileged user/network namespaces, Python 3, Git, GitHub CLI, Bubblewrap, and the standalone Codex binary. Bubblewrap is installed on this Pi. Run everything as `kopi`.

```sh
sudo apt-get install -y --no-install-recommends bubblewrap
```

GitHub authentication:

```sh
gh auth login --hostname github.com --git-protocol https --web
```

```sh
gh auth setup-git && gh auth status
```

Codex needs an existing **ChatGPT file login** at `$CODEX_HOME/auth.json` or `~/.codex/auth.json`. The worker uses an ephemeral copy; it never switches to an API key. Refreshed ChatGPT credentials are validated and atomically saved back only if the host login file has not changed. Renew revoked/expired authentication with `codex login`. Avoid logging in/out while a review is running.

The existing GitHub browser login has broad account scopes. Configured repository selection restricts this runner's behavior, not the token itself. A separately provisioned fine-grained token can restrict credentials to selected repositories: Contents read and Pull requests write for the chosen PR issue-comment endpoint. Organization policies may require separate authorization.

Verify GitHub access and actual filesystem/network namespace isolation without a model call:

```sh
./bin/run --check
```

## Manual use

Discovery only (also the default); persists the initial baseline:

```sh
./bin/run --discover-only
```

Generate reviews for existing PRs locally, without publishing:

```sh
./bin/run --dry-run --review-existing
```

Dry runs use **separate state under `var/dry-run/`**, so they cannot change the production baseline or publication queue. Successful output is printed and retained in SQLite; rerunning a dry run reuses cached output. Dry runs consume Codex usage.

After inspecting a dry run, run production once:

```sh
./bin/run --publish --once
```

This first production activation skips existing ready revisions. To intentionally review the backlog too, add `--review-existing`.

Status and failed-job retry:

```sh
./bin/run --status
```

```sh
./bin/run --publish --retry-failed
```

```sh
./bin/run --state-dir var/dry-run --status
```

Retries preserve already-generated output after a posting failure. Deterministic comment markers and publisher-ID checks reconcile accepted posts after crashes or lost HTTP responses. SQLite retains revision identities so unchanged commits do not duplicate reviews. Exhausted retries remain `failed` until explicitly retried. Binary file contents are omitted while remaining text changes are reviewed; oversized, symlink, and submodule changes are `skipped` for manual review.

## Scheduling

Monitoring is scheduled every 10 minutes in `kopi`'s crontab, using this entry:

```cron
*/10 * * * * TZ=Europe/Bratislava /home/kopi/automations/github-review/bin/run --publish 2>&1 | /usr/bin/logger -t github-review
```

The runner sets its working directory and PATH, uses absolute resolved executable paths, lowers CPU priority, and acquires `flock` through Python's `fcntl`. Cron must use the same HOME and any custom CODEX_HOME/GH_CONFIG_DIR used during setup. One shared worker lock excludes production and dry-run model work. Overlapping invocations exit successfully.

Watch the activity log:

```sh
tail -f /home/kopi/automations/github-review/var/review.log
```

Each cycle logs timestamps with the local UTC offset, start, repository PR counts, per-review outcomes, and a completion summary such as `0 reviews posted, 0 recovered, 0 generated, 0 attempted, 0 errors`. Recovered counts are previously accepted posts confirmed after a lost response, not new posts. Skipped overlapping runs, interruptions, and startup failures are also logged. The log rotates at 1 MB with three backups; stdout/stderr additionally go to syslog under `github-review`.

To stop scheduling, use `EDITOR=vim crontab -e` and remove only the `GITHUB PR REVIEWER` block.

## Review scope and isolation

To avoid executing PR code, Git produces a bounded merge-base diff plus complete current text of changed files. These are supplied to Codex as data. **The model does not browse unchanged files or run tests.** Every comment states this scope; incomplete output is rejected. This is a deliberate narrower source-context implementation of Plan 2, suitable for the Pi. A later trusted read-only source tool could expand context.

- Bare Git caches fetch exact base commits and GitHub pull refs, including fork heads. Git hooks, external diff/textconv, user/system Git configuration, submodule recursion, and dependency installation are disabled.
- Bubblewrap exposes a fresh home, system runtime, trusted worker/schema, and a copy of Codex authentication. Host home, GitHub credentials, Git caches, other automations, and host processes are absent.
- All network interfaces except private loopback are absent. A Unix-socket relay permits HTTPS CONNECT only to `chatgpt.com`, `auth.openai.com`, and `api.openai.com` on port 443, with public destination IP validation. Other destinations, including home-network addresses, are denied. TLS remains end-to-end.
- Shell execution, apps, hooks, browser/computer tools, remote plugins, and user/project instruction loading are disabled. PR files are never checked out into the worker.
- Command output, model result size, time, changed-file count, diff/context size, Git-cache disk use, and minimum free space are bounded. Process groups are killed on timeout/interruption; disposable homes are removed. Subsequent runs remove leftovers from interrupted workers.
- Limits are operational bounds, not a strict monetary cap. Polling disk usage can briefly overshoot the cache threshold while a fetch is writing.

The runner fails closed if isolation/authentication checks fail. It does not silently use the host environment or relax restrictions.

## State, logs, tests

`var/` has mode 0700; SQLite, output, and logs are private under a restrictive umask. `review.log` rotates at 1 MB with three backups. Logs contain repository/PR metadata and sanitized error categories, never raw subprocess output or private diffs. Duration is stored per revision. Generated bodies are retained for posting retries; old posted/superseded/skipped bodies and caches are cleaned after `retention_days` (default 30). Revision identities remain for deduplication. Dry-run and production storage limits apply separately.

```sh
python3 -m unittest discover -s tests -v
```

```sh
python3 -m py_compile review.py isolation.py
```

Tests mock GitHub and Codex calls; they use no network or model credits. `--check` additionally probes real OS isolation. Model availability is established by a successful dry run, not assumed from a model name.

Codex behavior follows the [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode) and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference), checked against this Pi's CLI help.
