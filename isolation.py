"""Bubblewrap worker and narrow HTTPS relay; no host network or home mounts."""
import contextlib
import ipaddress
import json
import os
from pathlib import Path
import resource
import select
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading

ALLOWED_HOSTS = frozenset({'chatgpt.com', 'auth.openai.com', 'api.openai.com'})


def relay(left, right):
    for sock in (left, right):
        sock.settimeout(30)
    while True:
        ready, _, _ = select.select([left, right], [], [], 60)
        if not ready:
            return
        for source in ready:
            data = source.recv(65536)
            if not data:
                return
            (right if source is left else left).sendall(data)


def destination(authority):
    host, sep, port = authority.lower().rpartition(':')
    if not sep or port != '443' or host not in ALLOWED_HOSTS:
        raise ValueError('destination denied')
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('non-public destination denied')
    return addresses


class Gateway(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(15)
        try:
            header = bytearray()
            while not header.endswith(b'\r\n\r\n'):
                chunk = self.request.recv(1)
                if not chunk or len(header) >= 8192:
                    return
                header.extend(chunk)
            method, authority, version = header.split(b'\r\n', 1)[0].decode('ascii').split()
            if method != 'CONNECT' or version not in ('HTTP/1.0', 'HTTP/1.1'):
                raise ValueError('CONNECT required')
            addresses = destination(authority)
            for family, kind, proto, _, address in addresses:
                remote = socket.socket(family, kind, proto)
                remote.settimeout(15)
                try:
                    remote.connect(address)
                    break
                except OSError:
                    remote.close()
            else:
                raise OSError('destination unavailable')
            with remote:
                self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                relay(self.request, remote)
        except (OSError, ValueError, UnicodeError):
            with contextlib.suppress(OSError):
                self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')


class UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    # Bound relay threads independently of model behavior.
    slots = threading.BoundedSemaphore(16)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Bridge(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            with socket.socket(socket.AF_UNIX) as remote:
                remote.connect('/relay.sock')
                relay(self.request, remote)
        except OSError:
            pass


def save_refreshed_auth(auth, original, copy):
    """Keep unattended refreshes, without overwriting a newer host login."""
    updated = copy.read_bytes()
    if updated == original or auth.read_bytes() != original:
        return
    before, after = json.loads(original), json.loads(updated)
    if (after.get('auth_mode') != 'chatgpt' or after.get('OPENAI_API_KEY')
            or not isinstance(after.get('tokens'), dict)
            or after['tokens'].get('account_id') != before['tokens'].get('account_id')
            or not all(isinstance(after['tokens'].get(k), str) and after['tokens'][k]
                       for k in ('access_token', 'refresh_token', 'id_token'))):
        raise RuntimeError('Invalid refreshed ChatGPT credentials')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=auth.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        if auth.read_bytes() == original:
            os.replace(temporary, auth)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def worker(directory, codex, auth, schema):
    """Expose only runtime files, a disposable home, and an allowlisted relay."""
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError('Bubblewrap missing: install bubblewrap')
    directory = Path(directory).resolve()
    home = directory / 'home'
    home.mkdir(mode=0o700)
    codex_home = home / '.codex'
    codex_home.mkdir(mode=0o700)
    auth = Path(auth)
    original = auth.read_bytes()
    credentials = json.loads(original)
    if credentials.get('auth_mode') != 'chatgpt' or not credentials.get('tokens'):
        raise RuntimeError('Existing Codex ChatGPT file login required')
    auth_copy = codex_home / 'auth.json'
    auth_copy.write_bytes(original)
    auth_copy.chmod(0o600)
    sock = directory / 'relay.sock'
    server = UnixServer(str(sock), Gateway)
    sock.chmod(0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = [bwrap, '--die-with-parent', '--new-session', '--unshare-all',
            '--cap-drop', 'ALL', '--clearenv', '--ro-bind', '/usr', '/usr']
    for path in ('/bin', '/sbin', '/lib', '/lib64'):
        if Path(path).is_symlink():
            args += ['--symlink', os.readlink(path), path]
        elif Path(path).exists():
            args += ['--ro-bind', path, path]
    args += ['--dir', '/etc', '--ro-bind', '/etc/ssl/certs', '/etc/ssl/certs',
             '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
             '--bind', str(home), '/home/worker',
             '--ro-bind', str(Path(codex).resolve()), '/codex',
             '--ro-bind', str(Path(__file__).resolve()), '/isolation.py',
             '--ro-bind', str(Path(schema).resolve()), '/schema.json',
             '--ro-bind', str(sock), '/relay.sock',
             '--dir', '/review', '--chdir', '/review',
             '--setenv', 'HOME', '/home/worker',
             '--setenv', 'CODEX_HOME', '/home/worker/.codex',
             '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LANG', 'C.UTF-8',
             '/usr/bin/python3', '/isolation.py']
    try:
        yield args, home
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        save_refreshed_auth(auth, original, auth_copy)


def main():
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.argv[1:] == ['--probe']:
        assert not Path('/home/kopi').exists()
        assert not Path('/etc/passwd').exists()
        assert not any(k in os.environ for k in ('GH_TOKEN', 'GITHUB_TOKEN', 'OPENAI_API_KEY'))
        assert socket.if_nameindex() == [(1, 'lo')]
        with socket.socket(socket.AF_UNIX) as remote:
            remote.settimeout(5)
            remote.connect('/relay.sock')
            remote.sendall(b'CONNECT github.com:443 HTTP/1.1\r\n\r\n')
            assert b'403 Forbidden' in remote.recv(1024)
        print('isolation probe passed')
        return 0
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Bridge)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = f'http://127.0.0.1:{server.server_address[1]}'
    os.environ.update(HTTPS_PROXY=proxy, HTTP_PROXY=proxy, https_proxy=proxy, http_proxy=proxy)
    try:
        return subprocess.call(sys.argv[1:])
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    sys.exit(main())
