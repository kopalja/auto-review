import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
        self.generator.generate.return_value = 'Validated body'

    def seed(self, pulls=None, backfill=True):
        review.discover(self.db, 'owner/repo', 'gpt-5.6-sol', [pr()] if pulls is None else pulls, backfill)
        return self.row()

    def row(self, head='a' * 40):
        return self.db.execute('SELECT * FROM revisions WHERE head=?', (head,)).fetchone()

    def process(self, dry=False):
        review.process(self.db, self.row(), self.gh, self.generator, self.cfg, 99, dry)


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
        review.discover(self.db, 'other/repo', 'gpt-5.6-sol', [pr()], False)
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
        review.discover(self.db, 'other/repo', 'gpt-5.6-sol', [pr()], True)
        self.process()
        row = review.next_job(self.db, ['owner/repo', 'other/repo'], set())
        self.assertEqual(row['repo'], 'other/repo')
        review.update(self.db, row, next_retry=review.time.time() + 100)
        row = review.next_job(self.db, ['owner/repo', 'other/repo'], set())
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
                cfg.write_text(json.dumps({'repositories': ['owner/repo']}))
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
                gen.generate.return_value = 'validated body'
                if scenario == 'recovered':
                    db = review.database(state / 'state.sqlite3')
                    review.discover(db, 'owner/repo', 'gpt-5.6-sol', [pr()], True)
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
            row = dict(repo='owner/repo', number=1, head='a' * 40, model='gpt-5.6-sol')
            @contextlib.contextmanager
            def fake_worker(directory, *args):
                home = Path(directory)
                (home / 'result.json').write_text(json.dumps({'complete': True, 'limitations': [], 'findings': []}))
                yield ['isolated-worker'], home
            with patch.object(gen, 'context', return_value=('review input', {})), patch('review.worker', fake_worker), patch('review.command') as command:
                gen.generate(row, pr())
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


if __name__ == '__main__':
    unittest.main()
