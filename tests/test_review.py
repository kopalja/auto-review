import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import isolation
import review


def pr(head='a' * 40, base='b' * 40, draft=False, state='open', number=1):
    return dict(number=number, head={'sha': head}, base={'sha': base}, draft=draft,
                state=state, title='Example', body='Description')


class ReviewerFixture(unittest.TestCase):
    def setUp(self):
        self.db = review.database(':memory:')
        self.addCleanup(self.db.close)
        self.cfg = review.DEFAULTS.copy()
        self.gh = Mock()
        self.gh.pull.return_value = pr()
        self.gh.reconcile.return_value = None
        self.gh.publish.return_value = 42
        self.generator = Mock()
        self.generator.context.return_value = ('review input', {})
        self.generator.generate.return_value = 'Validated body'

    def seed(self, pulls=None, backfill=True, reviewers=('codex',)):
        review.discover(self.db, 'owner/repo', 'gpt-5.6-sol', [pr()] if pulls is None else pulls, backfill, reviewers)
        return self.row()

    def row(self, head='a' * 40, reviewer='codex'):
        return self.db.execute("SELECT * FROM revisions WHERE repo='owner/repo' AND head=? AND reviewer=?", (head, reviewer)).fetchone()

    def rows(self, head='a' * 40):
        return self.db.execute("SELECT * FROM revisions WHERE repo='owner/repo' AND head=? ORDER BY reviewer DESC", (head,)).fetchall()

    def process(self, dry=False):
        return review.process(self.db, self.rows(), self.gh, self.generator, self.cfg, 99, dry)


class ReviewerTests(ReviewerFixture):
    def test_first_activation_and_new_commit(self):
        self.seed(backfill=False)
        self.assertEqual(self.row()['status'], 'baseline')
        self.seed([pr('c' * 40)], backfill=False)
        self.assertEqual(self.row('c' * 40)['status'], 'pending')
        self.assertEqual(self.row()['status'], 'baseline')

    def test_draft_ready_same_commit(self):
        self.seed([pr(draft=True)], backfill=False)
        self.assertIsNone(self.row())
        self.seed(backfill=False)
        self.assertEqual(self.row()['status'], 'pending')

    def test_baseline_per_repository(self):
        self.seed()
        review.discover(self.db, 'other/repo', 'gpt-5.6-sol', [pr()], False, ('codex',))
        self.assertEqual(self.db.execute("SELECT status FROM revisions WHERE repo='other/repo'").fetchone()[0], 'baseline')

    def test_unchanged_head_does_not_duplicate(self):
        self.seed()
        self.process()
        self.seed(backfill=False)
        self.assertEqual(self.row()['status'], 'posted')
        self.gh.publish.assert_called_once()

    def test_posting_failure_reuses_result(self):
        self.seed()
        self.gh.publish.side_effect = review.Failure('unavailable')
        self.process()
        self.assertEqual(self.row()['status'], 'generated')
        self.assertEqual(self.row()['body'], 'Validated body')
        self.gh.publish.side_effect = None
        self.process()
        self.generator.generate.assert_called_once()
        self.assertEqual(self.row()['status'], 'posted')

    def test_ambiguous_post_recovers_without_duplicate(self):
        self.seed()
        self.gh.publish.side_effect = review.Failure('lost response')
        self.process()
        self.gh.reconcile.return_value = 17
        self.process()
        self.gh.publish.assert_called_once()
        self.assertEqual(self.row()['publication_id'], 17)

    def test_stale_head_does_not_post(self):
        self.seed()
        self.gh.pull.side_effect = [pr(), pr('c' * 40)]
        self.process()
        self.gh.publish.assert_not_called()
        self.assertEqual(self.row()['status'], 'superseded')

    def test_stale_base_requeues_generation(self):
        self.seed()
        self.gh.pull.side_effect = [pr(), pr(base='c' * 40)]
        self.process()
        self.gh.publish.assert_not_called()
        self.assertEqual(self.row()['base'], 'c' * 40)
        self.assertEqual(self.row()['status'], 'pending')
        self.assertIsNone(self.row()['body'])

    def test_base_movement_does_not_repeat_posted_review(self):
        self.seed()
        self.process()
        self.seed([pr(base='c' * 40)], backfill=False)
        self.assertEqual(self.row()['status'], 'posted')

    def test_close_or_draft_before_publish(self):
        for changed in (pr(state='closed'), pr(draft=True)):
            with self.subTest(changed=changed):
                self.db.execute('DELETE FROM revisions')
                self.seed()
                self.gh.pull.side_effect = [pr(), changed]
                self.process()
                self.gh.publish.assert_not_called()

    def test_failure_backoff_and_exhaustion(self):
        self.seed()
        self.generator.generate.side_effect = review.Failure('timeout', retry_after=3600)
        for _ in range(3):
            self.process()
        self.assertEqual(self.row()['status'], 'failed')
        self.assertEqual(self.row()['generation_attempts'], 3)
        self.assertGreater(self.row()['next_retry'], review.time.time() + 3500)

    def test_publication_retries_preserve_body(self):
        self.seed()
        self.gh.publish.side_effect = review.Failure('offline')
        for _ in range(3):
            self.process()
        self.assertEqual(self.row()['status'], 'failed')
        self.assertEqual(self.row()['body'], 'Validated body')
        self.assertEqual(self.row()['publication_attempts'], 3)
        self.generator.generate.assert_called_once()

    def test_interrupted_generation_has_bounded_attempts(self):
        row = self.seed()
        review.update(self.db, row, generation_attempts=3)
        self.process()
        self.generator.generate.assert_not_called()
        self.assertEqual(self.row()['status'], 'failed')

    def test_oversized_is_skipped(self):
        self.seed()
        self.generator.generate.side_effect = review.Limited('oversized')
        self.process()
        self.assertEqual(self.row()['status'], 'skipped')
        self.gh.publish.assert_not_called()

    def test_dry_run_never_posts(self):
        self.seed()
        with contextlib.redirect_stdout(io.StringIO()):
            self.process(dry=True)
        self.gh.publish.assert_not_called()
        self.gh.reconcile.assert_not_called()
        self.assertEqual(self.row()['status'], 'generated')

    def test_pagination(self):
        gh = review.GitHub()
        gh.api = Mock(side_effect=[[pr(number=n) for n in range(1, 101)], [pr(number=101)]])
        self.assertEqual(len(gh.pulls('owner/repo')), 101)
        self.assertIn('page=2', gh.api.call_args.args[0])

    def test_marker_reconciliation_requires_publisher_identity(self):
        gh = review.GitHub()
        gh.pages = Mock(return_value=[{'id': 5, 'user': {'id': 1}, 'body': 'marker'},
                                     {'id': 6, 'user': {'id': 99}, 'body': 'marker'}])
        self.assertEqual(gh.reconcile('owner/repo', 1, 'marker', 99), 6)

    def test_bad_gh_json(self):
        with patch('review.command', return_value=b'not json'):
            with self.assertRaises(review.Failure):
                review.GitHub().api('user')

    def test_validate_rejects_bad_results(self):
        good = {'complete': True, 'limitations': [], 'findings': []}
        self.assertEqual(review.validate(good, {}), good)
        bad = [None, {}, good | {'complete': False}, good | {'extra': 1},
               good | {'findings': [{'severity': 'P1', 'file': 'x', 'line': 1, 'title': 'T', 'explanation': 'E'}]}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(review.Failure):
                review.validate(value, {})

    def test_line_validation_and_safe_formatting(self):
        row = self.seed()
        result = {'complete': True, 'limitations': [], 'findings': [dict(
            severity='P1', file='x.py', line=3, title='@all <img>', explanation='![bad](https://example.com)')]}
        with self.assertRaises(review.Failure):
            review.validate(result, {'x.py': 2})
        review.validate(result, {'x.py': 3})
        body = review.format_review(row, result)
        self.assertTrue(body.startswith('## 🤖 Generated by Codex\n'))
        self.assertNotIn('@all', body)
        self.assertNotIn('<img>', body)
        self.assertIn(review.marker(row), body)


class BoundaryTests(unittest.TestCase):
    def test_command_output_limit_and_timeout(self):
        with self.assertRaises(review.Limited):
            review.command([sys.executable, '-c', 'print("x"*10000)'], limit=100)
        with self.assertRaises(review.Failure):
            review.command([sys.executable, '-c', 'import time;time.sleep(10)'], timeout=.05)

    def test_command_errors_do_not_expose_stderr(self):
        with self.assertRaises(review.Failure) as caught:
            review.command([sys.executable, '-c', 'import sys;sys.stderr.write("SECRET");sys.exit(1)'])
        self.assertNotIn('SECRET', str(caught.exception))

    def test_lock_excludes_overlapping_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            with review.lock(Path(directory) / 'lock') as first:
                self.assertTrue(first)
                with review.lock(Path(directory) / 'lock') as second:
                    self.assertFalse(second)
            with review.lock(Path(directory) / 'lock') as third:
                self.assertTrue(third)

    def test_configuration_and_local_repo_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / 'monitored-repos' / 'checkout'
            checkout.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', 'git@github.com:owner/repo.git'], check=True)
            path = root / 'repos.json'
            path.write_text(json.dumps({'discover_local_repositories': True, 'repositories': []}))
            self.assertEqual(review.config(path)['repositories'], [{'name': 'owner/repo', 'model': 'gpt-6-astra'}])
            outside = root / 'outside'
            outside.mkdir()
            subprocess.run(['git', 'init', '-q', str(outside)], check=True)
            subprocess.run(['git', '-C', str(outside), 'remote', 'add', 'origin', 'https://github.com/owner/outside.git'], check=True)
            path.write_text(json.dumps({'discover_local_repositories': True, 'repositories': ['owner/outside']}))
            self.assertEqual(review.config(path)['repositories'], [{'name': 'owner/repo', 'model': 'gpt-6-astra'}])
            path.write_text(json.dumps({'repositories': ['../../bad']}))
            with self.assertRaises(review.Failure):
                review.config(path)

    def test_missing_monitored_directory_discovers_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'repos.json'
            path.write_text(json.dumps({'discover_local_repositories': True}))
            self.assertEqual(review.config(path)['repositories'], [])

    def test_network_allowlist_denies_lan_and_other_hosts(self):
        for authority in ('localhost:443', '192.168.1.1:443', 'github.com:443', 'chatgpt.com:80', 'chatgpt.com.evil:443'):
            with self.subTest(authority=authority), self.assertRaises(ValueError):
                isolation.destination(authority)
        with patch('isolation.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 443))]):
            with self.assertRaises(ValueError):
                isolation.destination('chatgpt.com:443')

    def test_fork_fetch_uses_target_pull_ref_and_merge_base(self):
        with tempfile.TemporaryDirectory() as directory:
            gen = review.Generator(review.DEFAULTS, Path(directory))
            interrupted = gen.cache / 'owner--repo'
            interrupted.mkdir()
            (interrupted / 'HEAD').write_text('ref: refs/heads/main\n')
            (interrupted / 'config.lock').touch()
            calls = []
            def fake(cache, *args, **kwargs):
                calls.append(args)
                return { 'init': b'', 'fetch': b'', 'rev-parse': b'a' * 40,
                         'merge-base': b'c' * 40, 'diff': b'' }[args[0]]
            gen.git_command = fake
            row = dict(repo='owner/repo', number=5, head='a' * 40, base='b' * 40)
            prompt, files = gen.context(row, pr())
            self.assertFalse((interrupted / 'config.lock').exists())
            self.assertIn(('init', '--bare'), calls)
            self.assertTrue(any('https://github.com/owner/repo.git' in x and '+refs/pull/5/head:refs/review/head' in x for x in calls))
            self.assertIn(('merge-base', 'b' * 40, 'a' * 40), calls)
            self.assertEqual(files, {})

    def test_new_file_content_is_not_duplicated_outside_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            gen = review.Generator(review.DEFAULTS, Path(directory))
            def fake(cache, *args, **kwargs):
                if args[0] in ('init', 'fetch'):
                    return b''
                if args[0] == 'rev-parse':
                    return b'a' * 40
                if args[0] == 'merge-base':
                    return b'c' * 40
                if args[0] == 'diff' and '--name-only' in args:
                    return b'new.py\0'
                if args[0] == 'diff' and '--numstat' in args:
                    return b'1\t0\tnew.py\n'
                if args[0] == 'diff':
                    return b'+unique new content\n'
                if args[0] == 'ls-tree':
                    return b'100644 blob hash\tnew.py\n' if args[1] == 'a' * 40 else b''
                if args[0] == 'show':
                    return b'unique new content\n'
                raise AssertionError(args)
            gen.git_command = fake
            row = dict(repo='owner/repo', number=5, head='a' * 40, base='b' * 40)
            prompt, files = gen.context(row, pr())
            payload = json.loads(prompt.split('UNTRUSTED REVIEW DATA:\n', 1)[1])
            self.assertEqual(payload['files']['new.py'], '(new file; full contents are included in the diff)')
            self.assertEqual(files, {'new.py': 1})

    def test_binary_files_are_omitted_from_review(self):
        with tempfile.TemporaryDirectory() as directory:
            gen = review.Generator(review.DEFAULTS, Path(directory))
            def fake(cache, *args, **kwargs):
                if args[0] in ('init', 'fetch'):
                    return b''
                if args[0] == 'rev-parse':
                    return b'a' * 40
                if args[0] == 'merge-base':
                    return b'c' * 40
                if args[0] == 'diff' and '--name-only' in args:
                    return b'image.png\0code.py\0'
                if args[0] == 'diff' and '--numstat' in args:
                    return b'-\t-\timage.png\x001\t0\tcode.py\x00'
                if args[0] == 'diff':
                    return b'Binary files differ\n+print("ok")\n'
                if args[0] == 'ls-tree':
                    return b'100644 blob hash\tcode.py\n' if args[1] == 'a' * 40 else b''
                if args[0] == 'show':
                    return b'print("ok")\n'
                raise AssertionError(args)
            gen.git_command = fake
            row = dict(repo='owner/repo', number=5, head='a' * 40, base='b' * 40)
            prompt, files = gen.context(row, pr())
            payload = json.loads(prompt.split('UNTRUSTED REVIEW DATA:\n', 1)[1])
            self.assertEqual(payload['omitted_binary_files'], ['image.png'])
            self.assertNotIn('image.png', payload['files'])
            self.assertEqual(files, {'code.py': 1})


class RecoveryTests(ReviewerFixture):
    def test_publication_recovery_survives_discovery_of_new_base(self):
        self.seed()
        self.gh.publish.side_effect = review.Failure('lost response')
        self.process()
        self.seed([pr(base='d' * 40)], backfill=False)
        self.assertEqual(self.row()['body'], 'Validated body')
        self.gh.reconcile.return_value = 42
        self.process()
        self.assertEqual(self.row()['status'], 'posted')
        self.generator.generate.assert_called_once()

    def test_publication_recovery_survives_closed_pr(self):
        self.seed()
        self.gh.publish.side_effect = review.Failure('lost response')
        self.process()
        self.seed([], backfill=False)
        self.assertEqual(self.row()['status'], 'generated')
        self.gh.reconcile.return_value = 42
        self.process()
        self.assertEqual(self.row()['status'], 'posted')

    def test_reopened_unreviewed_head_is_pending(self):
        self.seed()
        self.seed([], backfill=False)
        self.assertEqual(self.row()['status'], 'superseded')
        self.seed(backfill=False)
        self.assertEqual(self.row()['status'], 'pending')

    def test_fairness_and_retry_times(self):
        self.seed([pr(), pr('c' * 40, number=2)])
        review.discover(self.db, 'other/repo', 'gpt-5.6-sol', [pr()], True, ('codex',))
        self.process()
        [row] = review.next_job(self.db, ['owner/repo', 'other/repo'], set())
        self.assertEqual(row['repo'], 'other/repo')
        review.update(self.db, row, next_retry=review.time.time() + 100)
        [row] = review.next_job(self.db, ['owner/repo', 'other/repo'], set())
        self.assertEqual(row['number'], 2)


class RunnerTests(unittest.TestCase):
    def test_discovery_failure_does_not_stop_other_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'repos.json'
            cfg.write_text(json.dumps({'repositories': ['bad/repo', 'good/repo']}))
            gh = Mock()
            gh.api.return_value = {'id': 99}
            gh.pulls.side_effect = [review.Failure('offline'), [pr()]]
            with patch('review.GitHub', return_value=gh), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review.main(['--config', str(cfg), '--state-dir', str(root / 'state'), '--discover-only']), 1)
            db = review.database(root / 'state/state.sqlite3')
            self.addCleanup(db.close)
            self.assertEqual(db.execute('SELECT repo,status FROM revisions').fetchone()[:], ('good/repo', 'baseline'))

    def test_dry_run_state_is_separate_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'repos.json'
            cfg.write_text(json.dumps({'repositories': ['owner/repo']}))
            gh = Mock()
            gh.api.return_value = {'id': 99}
            gh.pulls.return_value = [pr()]
            gh.pull.return_value = pr()
            gen = Mock()
            gen.tmp = root / 'tmp'
            gen.cache = root / 'cache'
            gen.tmp.mkdir()
            gen.cache.mkdir()
            gen.context.return_value = ('review input', {})
            gen.generate.return_value = 'preview'
            with patch('review.GitHub', return_value=gh), patch('review.Generator', return_value=gen), contextlib.redirect_stdout(io.StringIO()):
                review.main(['--config', str(cfg), '--state-dir', str(root / 'state'), '--dry-run', '--review-existing'])
            gh.publish.assert_not_called()
            self.assertFalse((root / 'state/state.sqlite3').exists())
            self.assertTrue((root / 'state/dry-run/state.sqlite3').exists())

    def test_activity_log_counts_each_run(self):
        for scenario in ('idle', 'posted', 'recovered', 'failed'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg = root / 'repos.json'
                cfg.write_text(json.dumps({'repositories': ['owner/repo'], 'reviewers': ['codex']}))
                state = root / 'state'
                state.mkdir()
                gh = Mock()
                gh.api.return_value = {'id': 99}
                gh.pulls.return_value = [pr()]
                gh.pull.return_value = pr()
                gh.reconcile.return_value = None
                gh.publish.return_value = 42
                gen = Mock()
                gen.tmp, gen.cache = root / 'tmp', root / 'cache'
                gen.tmp.mkdir()
                gen.cache.mkdir()
                gen.context.return_value = ('review input', {})
                gen.generate.return_value = 'validated body'
                if scenario == 'recovered':
                    db = review.database(state / 'state.sqlite3')
                    review.discover(db, 'owner/repo', 'gpt-5.6-sol', [pr()], True, ('codex',))
                    row = db.execute('SELECT * FROM revisions').fetchone()
                    review.update(db, row, status='generated', body='saved body')
                    db.close()
                    gh.reconcile.return_value = 42
                if scenario == 'failed':
                    gen.generate.side_effect = review.Failure('model unavailable')
                argv = ['--config', str(cfg), '--state-dir', str(state), '--publish']
                if scenario != 'idle':
                    argv.append('--review-existing')
                with patch('review.GitHub', return_value=gh), patch('review.Generator', return_value=gen), contextlib.redirect_stdout(io.StringIO()):
                    code = review.main(argv)
                log = (state / 'review.log').read_text()
                self.assertRegex(log, r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4} INFO Run started:')
                self.assertIn(f"Run finished: {int(scenario == 'posted')} reviews posted, {int(scenario == 'recovered')} recovered", log)
                self.assertEqual(code, int(scenario == 'failed'))
                if scenario == 'idle':
                    self.assertIn('0 generated, 0 attempted, 0 errors', log)
                elif scenario == 'recovered':
                    gen.generate.assert_not_called()
                    gh.publish.assert_not_called()
                elif scenario == 'posted':
                    self.assertIn('1 generated, 1 attempted, 0 errors', log)

    def test_overlapping_run_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'repos.json'
            cfg.write_text(json.dumps({'repositories': []}))
            with review.lock(root / 'reviewer.lock'):
                self.assertEqual(review.main(['--config', str(cfg), '--state-dir', str(root)]), 0)
            self.assertIn('Run skipped: previous invocation is still active', (root / 'review.log').read_text())

    def test_startup_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'repos.json'
            cfg.write_text(json.dumps({'model': 'invalid'}))
            with self.assertRaises(review.Failure):
                review.main(['--config', str(cfg), '--state-dir', str(root)])
            self.assertIn('Run aborted: Unsupported model', (root / 'review.log').read_text())

    def test_api_parses_headers(self):
        with patch('review.command', return_value=b'HTTP/2.0 200 OK\nX-Test: yes\r\n\r\n{"id":42}'):
            self.assertEqual(review.GitHub().api('user'), {'id': 42})


class ReasoningTests(unittest.TestCase):
    def test_default_and_allowed_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'repos.json'
            path.write_text('{}')
            self.assertEqual(review.config(path)['reasoning_effort'], 'high')
            self.assertEqual(review.config(path)['model'], 'gpt-6-astra')
            for level in ('low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
                with self.subTest(level=level):
                    path.write_text(json.dumps({'reasoning_effort': level}))
                    self.assertEqual(review.config(path)['reasoning_effort'], level)

    def test_invalid_levels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'repos.json'
            for level in ('invalid', 'HIGH', '', None, True, 1, [], {}):
                with self.subTest(level=level):
                    path.write_text(json.dumps({'reasoning_effort': level}))
                    with self.assertRaisesRegex(review.Failure, 'reasoning_effort must be one of'):
                        review.config(path)

    def test_reasoning_is_passed_to_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = review.DEFAULTS | {'reasoning_effort': 'high'}
            gen = review.Generator(cfg, root)
            row = dict(repo='owner/repo', number=1, head='a' * 40, reviewer='codex', model='gpt-5.6-sol')
            @contextlib.contextmanager
            def fake_worker(directory, *args):
                home = Path(directory)
                (home / 'result.json').write_text(json.dumps({'complete': True, 'limitations': [], 'findings': []}))
                yield ['isolated-worker'], home
            with patch('review.worker', fake_worker), patch('review.command') as command:
                gen.generate(row, 'review input', {})
            args = command.call_args.args[0]
            index = args.index('model_reasoning_effort="high"')
            self.assertEqual(args[index - 1], '-c')
            self.assertIn('--ignore-user-config', args)


class AuthRefreshTests(unittest.TestCase):
    def test_refresh_persists_only_for_same_unchanged_host_login(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth, copy = root / 'auth.json', root / 'copy.json'
            before = {'auth_mode': 'chatgpt', 'tokens': {'account_id': 'account', 'access_token': 'old', 'refresh_token': 'old', 'id_token': 'old'}}
            after = {'auth_mode': 'chatgpt', 'tokens': {'account_id': 'account', 'access_token': 'new', 'refresh_token': 'new', 'id_token': 'new'}}
            original = json.dumps(before).encode()
            auth.write_bytes(original)
            copy.write_text(json.dumps(after))
            isolation.save_refreshed_auth(auth, original, copy)
            self.assertEqual(json.loads(auth.read_text()), after)
            self.assertEqual(auth.stat().st_mode & 0o777, 0o600)
            auth.write_text('newer host login')
            isolation.save_refreshed_auth(auth, original, copy)
            self.assertEqual(auth.read_text(), 'newer host login')

    def test_refresh_rejects_different_account_and_api_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth, copy = root / 'auth.json', root / 'copy.json'
            before = {'auth_mode': 'chatgpt', 'tokens': {'account_id': 'account'}}
            original = json.dumps(before).encode()
            auth.write_bytes(original)
            for after in ({'auth_mode': 'api', 'OPENAI_API_KEY': 'fake'},
                          {'auth_mode': 'chatgpt', 'tokens': {'account_id': 'other'}}):
                copy.write_text(json.dumps(after))
                with self.assertRaises(RuntimeError):
                    isolation.save_refreshed_auth(auth, original, copy)
                self.assertEqual(auth.read_bytes(), original)


class ClaudeTests(ReviewerFixture):
    def test_new_head_gets_one_row_per_reviewer(self):
        self.seed(reviewers=review.REVIEWERS)
        self.assertEqual({(r['reviewer'], r['model']) for r in self.rows()},
                         {('codex', 'gpt-5.6-sol'), ('claude', 'claude-opus-5-5')})

    def test_both_reviews_post_separate_comments(self):
        self.seed(reviewers=review.REVIEWERS)
        self.generator.generate.side_effect = lambda row, prompt, files: f"{row['reviewer']} body"
        self.gh.publish.side_effect = [1, 2]
        self.assertEqual(self.process(), {'codex': 'posted', 'claude': 'posted'})
        self.generator.context.assert_called_once()
        self.assertEqual([c.args[2] for c in self.gh.publish.call_args_list], ['codex body', 'claude body'])
        self.assertEqual({r['reviewer']: r['publication_id'] for r in self.rows()}, {'codex': 1, 'claude': 2})
        self.gh.pull.assert_called()
        self.assertEqual(self.gh.pull.call_count, 2)  # Once before generation, once before posting.

    def test_models_run_in_parallel(self):
        self.seed(reviewers=review.REVIEWERS)
        barrier = threading.Barrier(2, timeout=5)
        def generate(row, prompt, files):
            barrier.wait()  # Raises BrokenBarrierError if the calls were sequential.
            return 'body'
        self.generator.generate.side_effect = generate
        self.assertEqual(self.process(), {'codex': 'posted', 'claude': 'posted'})

    def test_one_reviewer_failing_does_not_block_the_other(self):
        self.seed(reviewers=review.REVIEWERS)
        def generate(row, prompt, files):
            if row['reviewer'] == 'claude':
                raise review.Failure('claude exited 1')
            return 'codex body'
        self.generator.generate.side_effect = generate
        self.assertEqual(self.process(), {'codex': 'posted'})
        self.gh.publish.assert_called_once()
        claude = self.row(reviewer='claude')
        self.assertEqual((claude['status'], claude['generation_attempts'], claude['error']),
                         ('pending', 1, 'claude exited 1'))
        self.assertEqual(self.row()['status'], 'posted')

    def test_context_limit_skips_every_reviewer(self):
        self.seed(reviewers=review.REVIEWERS)
        self.generator.context.side_effect = review.Limited('oversized')
        self.process()
        self.assertEqual({r['status'] for r in self.rows()}, {'skipped'})
        self.generator.generate.assert_not_called()

    def test_markers_are_distinct_and_codex_marker_is_unchanged(self):
        self.seed(reviewers=review.REVIEWERS)
        codex, claude = self.row(), self.row(reviewer='claude')
        legacy = review.hashlib.sha256(f"owner/repo:1:{'a' * 40}".encode()).hexdigest()
        self.assertEqual(review.marker(codex), f'<!-- github-review:{legacy} -->')
        self.assertNotEqual(review.marker(claude), review.marker(codex))

    def test_claude_comment_header(self):
        self.seed(reviewers=review.REVIEWERS)
        body = review.format_review(self.row(reviewer='claude'), {'complete': True, 'limitations': [], 'findings': []})
        self.assertTrue(body.startswith('## 🤖 Generated by Claude\n\n### Claude review · claude-opus-5-5\n'))

    def test_disabled_reviewer_is_not_scheduled(self):
        self.seed(reviewers=review.REVIEWERS)
        self.assertEqual([r['reviewer'] for r in review.next_job(self.db, ['owner/repo'], set())], ['codex', 'claude'])
        self.assertEqual([r['reviewer'] for r in review.next_job(self.db, ['owner/repo'], set(), ['codex'])], ['codex'])

    def test_known_heads_are_not_backfilled_for_claude(self):
        self.seed()
        self.process()
        self.seed([pr(), pr('c' * 40, number=2)], backfill=False, reviewers=review.REVIEWERS)
        self.assertEqual([r['reviewer'] for r in self.rows()], ['codex'])
        self.assertEqual(len(self.rows('c' * 40)), 2)


class MigrationTests(unittest.TestCase):
    def test_existing_state_becomes_codex_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.sqlite3'
            old = review.sqlite3.connect(path)
            old.executescript('''
            CREATE TABLE revisions(
              repo TEXT NOT NULL, number INTEGER NOT NULL, head TEXT NOT NULL, base TEXT NOT NULL,
              model TEXT NOT NULL, status TEXT NOT NULL, discovered REAL NOT NULL,
              generation_attempts INTEGER NOT NULL DEFAULT 0, publication_attempts INTEGER NOT NULL DEFAULT 0,
              next_retry REAL NOT NULL DEFAULT 0, body TEXT, publication_id INTEGER,
              error TEXT, duration REAL, updated REAL NOT NULL,
              PRIMARY KEY(repo, number, head));
            INSERT INTO revisions(repo,number,head,base,model,status,discovered,body,publication_id,updated)
              VALUES('owner/repo',1,'h','b','gpt-6-astra','posted',1,'body',7,2);
            ''')
            old.commit()
            old.close()
            for _ in range(2):  # Migration is idempotent.
                db = review.database(path)
                row = db.execute('SELECT * FROM revisions').fetchone()
                self.assertEqual((row['reviewer'], row['status'], row['body'], row['publication_id']),
                                 ('codex', 'posted', 'body', 7))
                db.close()


class ClaudeGeneratorTests(unittest.TestCase):
    def run_claude(self, output):
        with tempfile.TemporaryDirectory() as directory:
            gen = review.Generator(review.DEFAULTS | {'claude_reasoning_effort': 'max'}, Path(directory))
            row = dict(repo='owner/repo', number=1, head='a' * 40, reviewer='claude', model='claude-opus-5-5')
            @contextlib.contextmanager
            def fake_worker(directory, tool, *args):
                self.assertEqual(tool, 'claude')
                yield ['isolated-worker'], Path(directory)
            with patch('review.worker', fake_worker), patch('review.command', return_value=json.dumps(output).encode()) as command:
                body = gen.generate(row, 'review input', {})
            return body, command.call_args

    def test_claude_invocation_and_structured_output(self):
        output = {'type': 'result', 'subtype': 'success', 'is_error': False,
                  'structured_output': {'complete': True, 'limitations': [], 'findings': []}}
        body, call = self.run_claude(output)
        self.assertIn('Generated by Claude', body)
        args = call.args[0]
        self.assertEqual(args[args.index('--model') + 1], 'claude-opus-5-5')
        self.assertEqual(args[args.index('--effort') + 1], 'max')
        self.assertEqual(args[args.index('--tools') + 1], '')
        self.assertEqual(json.loads(args[args.index('--json-schema') + 1])['required'], ['complete', 'limitations', 'findings'])
        self.assertIn('--no-session-persistence', args)
        self.assertEqual(call.kwargs['data'], 'review input')

    def test_claude_errors_are_rejected(self):
        for output in ({'subtype': 'success', 'is_error': True, 'structured_output': {}},
                       {'subtype': 'error_max_turns', 'is_error': False},
                       {'subtype': 'success', 'is_error': False, 'structured_output': {'complete': False, 'limitations': [], 'findings': []}}):
            with self.subTest(output=output), self.assertRaises(review.Failure):
                self.run_claude(output)

    def test_cancel_stops_running_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            gen = review.Generator(review.DEFAULTS, Path(directory))
            gen.check_cancelled()
            gen.cancel()
            with self.assertRaises(review.Failure):
                review.command([sys.executable, '-c', 'import time;time.sleep(10)'], guard=gen.check_cancelled)

    def test_claude_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'repos.json'
            path.write_text('{}')
            self.assertEqual(review.config(path)['claude_reasoning_effort'], 'high')
            self.assertEqual(review.config(path)['reviewers'], ['codex', 'claude'])
            for bad in ({'claude_reasoning_effort': 'ultra'}, {'reviewers': []}, {'reviewers': ['gemini']},
                        {'reviewers': ['codex', 'codex']}, {'reviewers': 'codex'}):
                with self.subTest(bad=bad), self.assertRaises(review.Failure):
                    path.write_text(json.dumps(bad))
                    review.config(path)

    def test_claude_relay_allowlist(self):
        claude = isolation.ALLOWED_HOSTS['claude']
        for authority in ('chatgpt.com:443', 'claude.ai:443', 'api.anthropic.com:80'):
            with self.subTest(authority=authority), self.assertRaises(ValueError):
                isolation.destination(authority, claude)
        with patch('isolation.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('160.79.104.10', 443))]):
            isolation.destination('api.anthropic.com:443', claude)
        with self.assertRaises(ValueError):
            isolation.destination('api.anthropic.com:443')  # Codex relay does not reach Anthropic.

    def test_claude_refresh_persists_only_valid_subscription_login(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth, copy = root / 'credentials.json', root / 'copy.json'
            before = {'claudeAiOauth': {'accessToken': 'old', 'refreshToken': 'old', 'subscriptionType': 'max'}}
            after = {'claudeAiOauth': {'accessToken': 'new', 'refreshToken': 'new', 'subscriptionType': 'max'}}
            original = json.dumps(before).encode()
            auth.write_bytes(original)
            for bad in ({'claudeAiOauth': {'accessToken': 'new', 'refreshToken': ''}},
                        after | {'primaryApiKey': 'key'},
                        {'claudeAiOauth': after['claudeAiOauth'] | {'subscriptionType': 'pro'}}):
                copy.write_text(json.dumps(bad))
                with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                    isolation.save_refreshed_auth(auth, original, copy, 'claude')
                self.assertEqual(auth.read_bytes(), original)
            copy.write_text(json.dumps(after))
            isolation.save_refreshed_auth(auth, original, copy, 'claude')
            self.assertEqual(json.loads(auth.read_text()), after)


if __name__ == '__main__':
    unittest.main()
