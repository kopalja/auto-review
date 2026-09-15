#!/usr/bin/env python3
"""Sequential GitHub PR reviewer. Python standard library only."""
import argparse
import contextlib
import fcntl
import hashlib
import html
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time

from isolation import worker

ROOT = Path(__file__).resolve().parent
REPO = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z')
SHA = re.compile(r'[0-9a-f]{40}\Z')
REASONING_LEVELS = ('low', 'medium', 'high', 'xhigh', 'max', 'ultra')
DEFAULTS = dict(model='gpt-6-astra', reasoning_effort='high', max_reviews=3, review_timeout=900,
                max_attempts=3, retry_seconds=600, max_files=60,
                max_diff_bytes=200000, max_context_bytes=500000,
                max_cache_bytes=1073741824, min_free_bytes=536870912,
                retention_days=30)
LOG = logging.getLogger('github-review')


class Failure(Exception):
    def __init__(self, message, retry_after=0):
        super().__init__(message)
        self.retry_after = retry_after


class Limited(Failure):
    pass


def config(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - (set(DEFAULTS) | {'repositories', 'discover_local_repositories'}):
        raise Failure('Invalid configuration keys')
    settings = DEFAULTS | raw
    for key in DEFAULTS.keys() - {'model', 'reasoning_effort'}:
        if type(settings[key]) is not int or settings[key] <= 0:
            raise Failure(f'{key} must be a positive integer')
    if settings['model'] not in ('gpt-5.6-sol', 'gpt-6-astra'):
        raise Failure('Unsupported model')
    if settings['reasoning_effort'] not in REASONING_LEVELS:
        raise Failure('reasoning_effort must be one of: ' + ', '.join(REASONING_LEVELS))
    entries = settings.get('repositories', [])
    if not isinstance(entries, list):
        raise Failure('repositories must be a list')
    if type(settings.get('discover_local_repositories', False)) is not bool:
        raise Failure('discover_local_repositories must be boolean')
    discovered = None
    if settings.get('discover_local_repositories'):
        discovered = local_repositories(Path(path).resolve().parent / 'monitored-repos')
        explicit = set()
        for entry in entries:
            name = entry if isinstance(entry, str) else entry.get('name') if isinstance(entry, dict) else None
            if not isinstance(name, str):
                raise Failure('Invalid repository entry')
            explicit.add(name.lower())
        for name in discovered:
            if name not in explicit:
                entries = [*entries, name]
    seen = set()
    repos = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {'name': entry}
        if not isinstance(entry, dict) or set(entry) - {'name', 'enabled', 'model'}:
            raise Failure('Invalid repository entry')
        name = entry.get('name', '')
        if not isinstance(name, str) or not REPO.fullmatch(name) or '..' in name:
            raise Failure('Invalid owner/repository')
        name = name.lower()
        model = entry.get('model', settings['model'])
        if name in seen or model not in ('gpt-5.6-sol', 'gpt-6-astra'):
            raise Failure('Duplicate repository or unsupported model')
        if type(entry.get('enabled', True)) is not bool:
            raise Failure('enabled must be boolean')
        seen.add(name)
        if entry.get('enabled', True) and (discovered is None or name in discovered):
            repos.append({'name': name, 'model': model})
    settings['repositories'] = repos
    return settings


def local_repositories(directory):
    names = set()
    if not directory.exists():
        return []
    for child in sorted(directory.iterdir()):
        if child.is_symlink() or not child.is_dir() or not (child / '.git').exists():
            continue
        try:
            url = command([executable('git'), '-C', str(child), 'config', '--get', 'remote.origin.url']).decode().strip()
        except Failure:
            LOG.warning('Skipping checkout without a readable origin: %s', child.name)
            continue
        match = re.fullmatch(r'(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?', url)
        if not match:
            LOG.warning('Skipping checkout without a recognized GitHub origin: %s', child.name)
            continue
        names.add(match[1].lower())
    return sorted(names)


def executable(name):
    path = shutil.which(name)
    if not path:
        raise Failure(f'Missing executable: {name}')
    return str(Path(path).absolute())


def command(args, *, data=None, env=None, timeout=120, limit=2_000_000, guard=None):
    """Bound captured output, time, and the entire subprocess group lifecycle."""
    with tempfile.TemporaryFile() as stdin:
        if data is not None:
            stdin.write(data if isinstance(data, bytes) else data.encode())
        stdin.seek(0)
        proc = subprocess.Popen(args, stdin=stdin, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, start_new_session=True)
        stdout, stderr = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ, stdout)
                selector.register(proc.stderr, selectors.EVENT_READ, stderr)
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        raise Failure('Command timed out')
                    if guard:
                        guard()
                    for key, _ in selector.select(0.2):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            key.data.extend(chunk)
                            if len(stdout) + len(stderr) > limit:
                                raise Limited('Command output exceeds limit')
                proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            if proc.returncode:
                # Never surface arbitrary stderr (may contain credentials/private source).
                delay = 0
                match = re.search(rb'retry-after:\s*(\d+)', stderr + stdout, re.I)
                if match:
                    delay = int(match[1])
                if b'rate limit' in stderr.lower() or b'429' in stderr:
                    delay = max(delay, 3600)
                reset = re.search(rb'x-ratelimit-reset:\s*(\d+)', stdout, re.I)
                remaining = re.search(rb'x-ratelimit-remaining:\s*0\b', stdout, re.I)
                if reset and remaining:
                    delay = max(delay, int(reset[1]) - int(time.time()) + 1)
                raise Failure(f'{Path(args[0]).name} exited {proc.returncode}', delay)
            return bytes(stdout)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()


class GitHub:
    def __init__(self):
        self.gh = executable('gh')
        self.env = {k: v for k, v in os.environ.items()
                    if k in ('HOME', 'PATH', 'GH_TOKEN', 'GITHUB_TOKEN', 'GH_CONFIG_DIR',
                             'XDG_CONFIG_HOME', 'SSL_CERT_FILE', 'SSL_CERT_DIR')}
        self.env.update(GH_PROMPT_DISABLED='1', GH_PAGER='cat', NO_COLOR='1')

    def api(self, endpoint, payload=None):
        args = [self.gh, 'api', '--hostname', 'github.com', '--include', endpoint]
        if payload is not None:
            args += ['--method', 'POST', '--input', '-']
        try:
            output = command(args, env=self.env,
                             data=None if payload is None else json.dumps(payload))
            parts = re.split(rb'\r?\n\r?\n', output, maxsplit=1)
            if len(parts) != 2 or not parts[0].startswith(b'HTTP/'):
                raise Failure('Missing GitHub response headers')
            return json.loads(parts[1])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Failure('Invalid gh JSON output') from exc

    def pages(self, endpoint):
        result = []
        for page in range(1, 10001):
            rows = self.api(f'{endpoint}{"&" if "?" in endpoint else "?"}per_page=100&page={page}')
            if not isinstance(rows, list):
                raise Failure('Expected paginated GitHub list')
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise Failure('GitHub pagination limit exceeded')

    def pulls(self, repo):
        return self.pages(f'repos/{repo}/pulls?state=open&sort=created&direction=asc')

    def pull(self, repo, number):
        return self.api(f'repos/{repo}/pulls/{number}')

    def reconcile(self, repo, number, marker, user_id):
        for comment in self.pages(f'repos/{repo}/issues/{number}/comments'):
            if comment.get('user', {}).get('id') == user_id and marker in (comment.get('body') or ''):
                return comment['id']
        return None

    def publish(self, repo, number, body):
        return self.api(f'repos/{repo}/issues/{number}/comments', {'body': body})['id']


def database(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript('''
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS repositories(name TEXT PRIMARY KEY, last_served REAL NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS revisions(
      repo TEXT NOT NULL, number INTEGER NOT NULL, head TEXT NOT NULL, base TEXT NOT NULL,
      model TEXT NOT NULL, status TEXT NOT NULL, discovered REAL NOT NULL,
      generation_attempts INTEGER NOT NULL DEFAULT 0, publication_attempts INTEGER NOT NULL DEFAULT 0,
      next_retry REAL NOT NULL DEFAULT 0, body TEXT, publication_id INTEGER,
      error TEXT, duration REAL, updated REAL NOT NULL,
      PRIMARY KEY(repo, number, head));
    ''')
    return db


def identity(pr):
    try:
        number, head, base = pr['number'], pr['head']['sha'], pr['base']['sha']
        if type(number) is not int or number <= 0 or not SHA.fullmatch(head) or not SHA.fullmatch(base):
            raise ValueError()
        return number, head, base
    except (KeyError, TypeError, ValueError) as exc:
        raise Failure('Invalid GitHub PR identity') from exc


def discover(db, repo, model, pulls, backfill=False):
    now = time.time()
    # Validate all entries before committing a baseline.
    identities = [(pr, identity(pr)) for pr in pulls]
    first = not db.execute('SELECT 1 FROM repositories WHERE name=?', (repo,)).fetchone()
    with db:
        db.execute('INSERT OR IGNORE INTO repositories(name) VALUES(?)', (repo,))
        active = set()
        for pr, (number, head, base) in identities:
            active.add((number, head))
            if pr.get('draft', False):
                continue  # Drafts are never baselined, so ready transitions are eligible.
            status = 'baseline' if first and not backfill else 'pending'
            db.execute('INSERT OR IGNORE INTO revisions(repo,number,head,base,model,status,discovered,updated) VALUES(?,?,?,?,?,?,?,?)',
                       (repo, number, head, base, model, status, now, now))
            if backfill:
                db.execute("UPDATE revisions SET status='pending',updated=? WHERE repo=? AND number=? AND head=? AND status='baseline'",
                           (now, repo, number, head))
            db.execute("UPDATE revisions SET status='pending',base=?,model=?,generation_attempts=0,publication_attempts=0,next_retry=0,error=NULL,updated=? WHERE repo=? AND number=? AND head=? AND status='superseded'",
                       (base, model, now, repo, number, head))
            # Base changes update unfinished work only; completed heads never re-review.
            db.execute("UPDATE revisions SET base=?,status='pending',body=NULL,generation_attempts=0,publication_attempts=0,next_retry=0,error=NULL,updated=? WHERE repo=? AND number=? AND head=? AND base!=? AND body IS NULL AND status IN ('pending','failed')",
                       (base, now, repo, number, head, base))
        # Generated bodies must reach reconciliation even if the PR changed or closed.
        for row in db.execute("SELECT number,head FROM revisions WHERE repo=? AND body IS NULL AND status IN ('pending','failed')", (repo,)).fetchall():
            if (row['number'], row['head']) not in active:
                db.execute("UPDATE revisions SET status='superseded',body=NULL,updated=? WHERE repo=? AND number=? AND head=?",
                           (now, repo, row['number'], row['head']))


def update(db, row, **fields):
    fields['updated'] = time.time()
    with db:
        db.execute(f'UPDATE revisions SET {",".join(k+"=?" for k in fields)} WHERE repo=? AND number=? AND head=?',
                   (*fields.values(), row['repo'], row['number'], row['head']))


def marker(row):
    digest = hashlib.sha256(f"{row['repo']}:{row['number']}:{row['head']}".encode()).hexdigest()
    return f'<!-- github-review:{digest} -->'


def validate(result, files):
    if not isinstance(result, dict) or set(result) != {'complete', 'limitations', 'findings'}:
        raise Failure('Invalid review structure')
    if result['complete'] is not True:
        raise Failure('Review incomplete; no comment published')
    if not isinstance(result['limitations'], list) or any(not isinstance(x, str) or len(x) > 2000 for x in result['limitations']):
        raise Failure('Invalid review limitations')
    findings = result['findings']
    if not isinstance(findings, list) or len(findings) > 50 or len(result['limitations']) > 20:
        raise Failure('Invalid findings count')
    for f in findings:
        if not isinstance(f, dict) or set(f) != {'severity', 'file', 'line', 'title', 'explanation'}:
            raise Failure('Invalid finding')
        if not all(isinstance(f[k], str) and 0 < len(f[k]) <= 4000 for k in ('severity', 'file', 'title', 'explanation')):
            raise Failure('Invalid finding text')
        if f['severity'] not in ('P0', 'P1', 'P2', 'P3') or f['file'] not in files:
            raise Failure('Invalid finding location/severity')
        if type(f['line']) is not int or not 1 <= f['line'] <= files[f['file']]:
            raise Failure('Invalid finding line')
    return result


def plain(value):
    # Treat model text as text, preventing hidden markers, mentions, images and links.
    value = html.escape(value).replace('@', '@\u200b')
    return re.sub(r'([\\`*_{}\[\]()#+.!|>~-])', r'\\\1', value)


def format_review(row, result):
    lines = [f"### Codex review · {row['model']}", f"Commit: `{row['head']}`", '',
             'Scope: merge-base diff and full changed text files; no tests executed.', '']
    if not result['findings']:
        lines.append('No actionable findings in the reviewed scope.')
    for f in result['findings']:
        lines += [f"- **{f['severity']}: {plain(f['title'])}** — {plain(f['file'])}:{f['line']}",
                  f"  {plain(f['explanation'])}"]
    if result['limitations']:
        lines += ['', '**Limitations**'] + ['- ' + plain(x) for x in result['limitations']]
    lines += ['', marker(row)]
    body = '\n'.join(lines)
    if len(body.encode()) > 60000:
        raise Failure('Review comment too large')
    return body


class Generator:
    def __init__(self, settings, state):
        self.cfg, self.state = settings, state
        self.git, self.codex = executable('git'), executable('codex')
        self.auth = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'
        self.gh = executable('gh')
        self.cache = state / 'cache'
        self.cache.mkdir(exist_ok=True)
        self.tmp = state / 'tmp'
        self.tmp.mkdir(exist_ok=True)

    def check_disk(self):
        size = sum(p.stat().st_size for p in self.cache.rglob('*') if p.is_file())
        if size > self.cfg['max_cache_bytes']:
            raise Limited('Repository cache exceeds disk limit')
        if shutil.disk_usage(self.state).free < self.cfg['min_free_bytes']:
            raise Limited('Insufficient free disk space')

    def git_command(self, cache, *args, limit=2_000_000):
        env = {k: v for k, v in os.environ.items() if k in
               ('HOME', 'PATH', 'GH_TOKEN', 'GITHUB_TOKEN', 'GH_CONFIG_DIR', 'XDG_CONFIG_HOME')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                   GIT_TERMINAL_PROMPT='0', GH_PROMPT_DISABLED='1', GIT_LFS_SKIP_SMUDGE='1')
        return command([self.git, '-c', 'core.hooksPath=/dev/null', '-c', 'protocol.file.allow=never',
                        '-c', 'protocol.ext.allow=never', '-c', 'credential.helper=',
                        '-c', f'credential.helper=!{shlex.quote(self.gh)} auth git-credential',
                        '-c', 'core.fsmonitor=false', '-c', 'gc.auto=0', '-C', str(cache), *args],
                       env=env, timeout=180, limit=limit, guard=self.check_disk)

    def context(self, row, pr):
        cache = self.cache / row['repo'].replace('/', '--')
        try:
            # A killed init/fetch can leave an unusable cache or stale Git locks.
            # The runner lock guarantees that no other managed Git writer is active.
            if cache.exists() and (not (cache / 'HEAD').is_file() or any(cache.rglob('*.lock'))):
                shutil.rmtree(cache)
            self.check_disk()
            if not cache.exists():
                cache.mkdir()
                self.git_command(cache, 'init', '--bare')
            url = f"https://github.com/{row['repo']}.git"
            # GitHub pull refs cover fork heads; no fork URL or untrusted ref is executed.
            self.git_command(cache, 'fetch', '--no-tags', '--no-recurse-submodules', url,
                             row['base'], f"+refs/pull/{row['number']}/head:refs/review/head")
            os.utime(cache, None)
            actual = self.git_command(cache, 'rev-parse', 'refs/review/head').decode().strip()
            if actual != row['head']:
                self.git_command(cache, 'fetch', '--no-tags', '--no-recurse-submodules', url, row['head'])
            ancestor = self.git_command(cache, 'merge-base', row['base'], row['head']).decode().strip()
            diff = self.git_command(cache, 'diff', '--no-ext-diff', '--no-textconv', '--no-renames',
                                    ancestor, row['head'], '--', limit=self.cfg['max_diff_bytes'])
            names = self.git_command(cache, 'diff', '--name-only', '-z', '--no-renames', ancestor, row['head'], '--')
            paths = [x.decode('utf-8') for x in names.split(b'\0') if x]
            if len(paths) > self.cfg['max_files']:
                raise Limited('Too many changed files')
            stats = self.git_command(cache, 'diff', '--numstat', '--no-renames', ancestor, row['head'], '--')
            if any(line.startswith(b'-\t-\t') for line in stats.splitlines()):
                raise Limited('Binary changes require manual review')
            context, files = {}, {}
            used = len(diff)
            for path in paths:
                mode = self.git_command(cache, 'ls-tree', row['head'], '--', path)
                if not mode:
                    old_mode = self.git_command(cache, 'ls-tree', ancestor, '--', path)
                    if not old_mode.startswith((b'100644 ', b'100755 ')):
                        raise Limited('Deleted symlink or submodule requires manual review')
                    context[path] = '(deleted; see diff)'
                    files[path] = 1
                    continue
                if not mode.startswith((b'100644 ', b'100755 ')):
                    raise Limited('Symlink or submodule changes require manual review')
                text = self.git_command(cache, 'show', f"{row['head']}:{path}", limit=self.cfg['max_context_bytes'])
                used += len(text)
                if used > self.cfg['max_context_bytes'] or b'\0' in text:
                    raise Limited('Source context exceeds limit or contains binary data')
                context[path] = text.decode('utf-8')
                files[path] = max(1, len(context[path].splitlines()))
            payload = {'repository': row['repo'], 'number': row['number'], 'head': row['head'],
                       'base': row['base'], 'merge_base': ancestor, 'title': pr.get('title', ''),
                       'description': pr.get('body') or '', 'diff': diff.decode('utf-8'), 'files': context}
            prompt = (ROOT / 'review-prompt.md').read_text() + '\nUNTRUSTED REVIEW DATA:\n' + json.dumps(payload)
            if len(prompt.encode()) > self.cfg['max_context_bytes'] * 2:
                raise Limited('Serialized review input exceeds limit')
            return prompt, files
        except (Limited, UnicodeError):
            shutil.rmtree(cache, ignore_errors=True)
            raise Limited('Review exceeds text/file/disk limits; manual review required')

    def generate(self, row, pr):
        prompt, files = self.context(row, pr)
        with tempfile.TemporaryDirectory(dir=self.tmp) as directory:
            with worker(directory, self.codex, self.auth, ROOT / 'review-schema.json') as (args, home):
                args += ['/codex', 'exec', '--ignore-user-config', '--ignore-rules',
                         '--skip-git-repo-check', '--ephemeral', '--sandbox', 'read-only',
                         '--model', row['model'], '--output-schema', '/schema.json',
                         '-c', f'model_reasoning_effort="{self.cfg["reasoning_effort"]}"',
                         '--output-last-message', '/home/worker/result.json', '--color', 'never',
                         '-c', 'approval_policy="never"', '-c', 'forced_login_method="chatgpt"',
                         '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
                         '-c', 'features.shell_tool=false', '-c', 'features.unified_exec=false',
                         '-c', 'features.apps=false', '-c', 'features.multi_agent=false',
                         '-c', 'features.shell_snapshot=false',
                         '-c', 'features.hooks=false', '-c', 'features.remote_plugin=false',
                         '-c', 'features.browser_use=false', '-c', 'features.computer_use=false',
                         '-c', 'features.image_generation=false', '-c', 'features.code_mode_host=false',
                         '-c', 'features.skill_search=false', '-c', 'features.tool_suggest=false', '-']
                command(args, data=prompt, env={'PATH': '/usr/bin:/bin'},
                        timeout=self.cfg['review_timeout'], limit=1_000_000)
                result_path = home / 'result.json'
                if not result_path.is_file() or result_path.stat().st_size > 100000:
                    raise Failure('Missing or oversized structured output')
                try:
                    result = json.loads(result_path.read_text())
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise Failure('Malformed structured review') from exc
                return format_review(row, validate(result, files))

    def probe(self):
        with tempfile.TemporaryDirectory(dir=self.tmp) as directory:
            with worker(directory, self.codex, self.auth, ROOT / 'review-schema.json') as (args, _):
                return command(args + ['--probe'], env={'PATH': '/usr/bin:/bin'}).decode().strip()


def process(db, row, gh, generator, cfg, user_id, dry_run=False):
    started = time.monotonic()
    stage = 'publication' if row['body'] else 'generation'
    try:
        # Reconcile before checking current PR state, including after a push or close.
        if row['body'] and not dry_run:
            published = gh.reconcile(row['repo'], row['number'], marker(row), user_id)
            if published:
                update(db, row, status='posted', publication_id=published, error=None)
                return 'recovered'
        pr = gh.pull(row['repo'], row['number'])
        _, head, base = identity(pr)
        if head != row['head'] or pr.get('state') != 'open':
            update(db, row, status='superseded', body=None)
            return
        if pr.get('draft'):
            update(db, row, next_retry=time.time() + cfg['retry_seconds'])
            return
        body = row['body']
        if base != row['base']:
            update(db, row, base=base, body=None, status='pending', generation_attempts=0,
                   publication_attempts=0, next_retry=0)
            return  # Next invocation reviews the new base, with the same head identity.
        if body is None:
            if row['generation_attempts'] >= cfg['max_attempts']:
                update(db, row, status='failed', error='Interrupted generation exhausted retries')
                return
            count = row['generation_attempts'] + 1
            update(db, row, generation_attempts=count)
            body = generator.generate(row, pr)
            update(db, row, body=body, status='generated', error=None, next_retry=0)
        if dry_run:
            print(body)
            return
        stage = 'publication'
        if row['publication_attempts'] >= cfg['max_attempts']:
            update(db, row, status='failed', error='Interrupted publication exhausted retries')
            return
        # A posting failure must not rerun generation, even across process restarts.
        update(db, row, publication_attempts=row['publication_attempts'] + 1)
        pr = gh.pull(row['repo'], row['number'])
        _, head, base = identity(pr)
        if head != row['head'] or pr.get('state') != 'open':
            update(db, row, status='superseded', body=None)
            return
        if pr.get('draft'):
            update(db, row, next_retry=time.time() + cfg['retry_seconds'])
            return
        if base != row['base']:
            update(db, row, base=base, body=None, status='pending', generation_attempts=0,
                   publication_attempts=0, next_retry=0)
            return
        published = gh.reconcile(row['repo'], row['number'], marker(row), user_id)
        outcome = 'recovered'
        if published is None:
            published = gh.publish(row['repo'], row['number'], body)
            outcome = 'posted'
        update(db, row, status='posted', publication_id=published, error=None, next_retry=0)
        return outcome
    except Limited as exc:
        update(db, row, status='skipped', error=str(exc))
    except (Failure, OSError, RuntimeError, ValueError, KeyError) as exc:
        current = db.execute('SELECT * FROM revisions WHERE repo=? AND number=? AND head=?',
                             (row['repo'], row['number'], row['head'])).fetchone()
        count = max(1, current[stage + '_attempts'])
        # Count failures that occurred before starting the stage too.
        if current[stage + '_attempts'] == row[stage + '_attempts']:
            count = current[stage + '_attempts'] + 1
        status = 'failed' if count >= cfg['max_attempts'] else ('generated' if current['body'] else 'pending')
        delay = max(cfg['retry_seconds'] * 2 ** min(count - 1, 8), getattr(exc, 'retry_after', 0))
        error = str(exc) if isinstance(exc, Failure) else type(exc).__name__
        update(db, row, status=status, error=error, next_retry=time.time() + delay,
               **{stage + '_attempts': count})
        LOG.error('%s #%s %s: %s', row['repo'], row['number'], stage, error)
    finally:
        update(db, row, duration=time.monotonic() - started)
        with db:
            db.execute('UPDATE repositories SET last_served=? WHERE name=?', (time.time(), row['repo']))


@contextlib.contextmanager
def lock(path):
    with open(path, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def next_job(db, available, processed):
    rows = db.execute("SELECT v.* FROM revisions v JOIN repositories r ON r.name=v.repo WHERE v.status IN ('pending','generated') AND v.next_retry<=? ORDER BY r.last_served,v.discovered,v.number", (time.time(),)).fetchall()
    return next((r for r in rows if r['repo'] in available and (r['repo'], r['number'], r['head']) not in processed), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'repos.json')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'var')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--discover-only', action='store_true')
    modes.add_argument('--dry-run', action='store_true', help='Generate locally in separate dry-run state; never post')
    modes.add_argument('--publish', action='store_true', help='Discover, generate and publish')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--check', action='store_true', help='Check GitHub login and OS isolation without model calls')
    parser.add_argument('--once', action='store_true', help='One invocation (also the default)')
    parser.add_argument('--review-existing', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    state = args.state_dir.resolve()
    if args.dry_run:
        state /= 'dry-run'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = RotatingFileHandler(state / 'review.log', maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                          datefmt='%Y-%m-%dT%H:%M:%S%z'))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    try:
        cfg = config(args.config)
        with lock(state / 'reviewer.lock') as acquired:
            if not acquired:
                LOG.info('Run skipped: previous invocation is still active')
                return 0
            # Shared lock also excludes dry-run model work from production runs.
            with lock(args.state_dir.resolve() / 'worker.lock') as worker_acquired:
                if not worker_acquired:
                    LOG.info('Run skipped: another review worker is still active')
                    return 0
                return run(args, cfg, state)
    except KeyboardInterrupt:
        LOG.warning('Run interrupted')
        raise
    except Exception as exc:
        LOG.error('Run aborted: %s', str(exc) if isinstance(exc, Failure) else type(exc).__name__)
        raise
    finally:
        LOG.removeHandler(handler)
        handler.close()


def run(args, cfg, state):
    started = time.monotonic()
    posted = recovered = attempted = generated = 0
    if not args.status and not args.check:
        mode = 'publish' if args.publish else 'dry-run' if args.dry_run else 'discovery'
        LOG.info('Run started: mode=%s repositories=%d reasoning_effort=%s',
                 mode, len(cfg['repositories']), cfg['reasoning_effort'])
    db = database(state / 'state.sqlite3')
    try:
        if args.status:
            for row in db.execute('SELECT repo,number,head,status,generation_attempts,publication_attempts,error FROM revisions ORDER BY discovered'):
                print(json.dumps(dict(row)))
            return 0
        gh = GitHub()
        user_id = gh.api('user')['id']
        if args.check:
            print(Generator(cfg, state).probe())
            for repo in cfg['repositories']:
                gh.api(f"repos/{repo['name']}")
            print('GitHub access passed; model availability is checked by a dry run')
            return 0
        if not cfg['repositories']:
            raise Failure('Add repositories to repos.json before running')
        generator = None
        if args.dry_run or args.publish:
            generator = Generator(cfg, state)
            generator.probe()  # Fail closed before discovery/state changes.
            for path in generator.tmp.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
            cutoff = time.time() - cfg['retention_days'] * 86400
            for path in generator.cache.iterdir():
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
            with db:
                db.execute("UPDATE revisions SET body=NULL WHERE status IN ('posted','superseded','skipped') AND updated<?", (cutoff,))
        available = []
        failures = 0
        for repo in cfg['repositories']:
            try:
                pulls = gh.pulls(repo['name'])
                discover(db, repo['name'], repo['model'], pulls, args.review_existing)
                LOG.info('Repository %s: %d open PRs', repo['name'], len(pulls))
                available.append(repo['name'])
            except (Failure, KeyError, ValueError, OSError) as exc:
                LOG.error('Discovery failed for %s: %s', repo['name'], type(exc).__name__)
                failures += 1
        if args.retry_failed:
            with db:
                for repo in available:
                    db.execute("UPDATE revisions SET status=CASE WHEN body IS NULL THEN 'pending' ELSE 'generated' END,generation_attempts=0,publication_attempts=0,next_retry=0,error=NULL WHERE repo=? AND status='failed'", (repo,))
        if generator:
            processed = set()
            for _ in range(cfg['max_reviews']):
                row = next_job(db, available, processed)
                if row is None:
                    break
                processed.add((row['repo'], row['number'], row['head']))
                attempted += 1
                LOG.info('Review started: %s #%s head=%s', row['repo'], row['number'], row['head'][:12])
                outcome = process(db, row, gh, generator, cfg, user_id, args.dry_run)
                posted += outcome == 'posted'
                recovered += outcome == 'recovered'
                current = db.execute('SELECT status,error,body,duration FROM revisions WHERE repo=? AND number=? AND head=?',
                                     (row['repo'], row['number'], row['head'])).fetchone()
                generated += row['body'] is None and current['body'] is not None
                LOG.info('Review finished: %s #%s status=%s outcome=%s duration=%.1fs',
                         row['repo'], row['number'], current['status'], outcome or current['status'], current['duration'])
                if current['error']:
                    failures += 1
        totals = dict(db.execute('SELECT status,COUNT(*) FROM revisions GROUP BY status').fetchall())
        LOG.info('Run finished: %d reviews posted, %d recovered, %d generated, %d attempted, %d errors; duration=%.1fs; state=%s',
                 posted, recovered, generated, attempted, failures, time.monotonic() - started, json.dumps(totals, sort_keys=True))
        print(json.dumps(totals))
        return 1 if failures else 0
    finally:
        db.close()


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except (Failure, OSError, RuntimeError, ValueError) as exc:
        print(f'github-review: {exc if isinstance(exc, Failure) else type(exc).__name__}', file=__import__('sys').stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
