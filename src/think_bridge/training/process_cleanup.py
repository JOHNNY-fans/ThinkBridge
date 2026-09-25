"""Bounded shutdown of subprocess groups owned by one training invocation."""

from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def cleanup_message(message):
    # Ctrl+C also stops tee. A closed log pipe must never stop TERM/KILL.
    try:
        print(message, flush=True)
    except OSError:
        pass


def spawn_owned_process(command, *, env=None, **kwargs):
    """Mark this subtree, including workers that later detach/reparent on Linux.

    Each nested launcher adds its own unique key, preserving ancestor keys.
    Thus an outer launcher can reclaim its entire tree without an inner one
    mistaking its parent or a neighbouring run for one of its workers.
    """
    environment = dict(os.environ if env is None else env)
    owner_tag = "THINK_BRIDGE_PROCESS_OWNER_" + uuid.uuid4().hex
    environment[owner_tag] = "1"
    process = subprocess.Popen(
        command, env=environment, start_new_session=True, **kwargs
    )
    process._think_bridge_owner_tag = owner_tag
    return process


@contextmanager
def termination_signals():
    """Turn termination into exceptions so driver finally blocks run."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def uninterrupted_cleanup():
    """Repeated Ctrl+C must not interrupt the bounded TERM/KILL sequence."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can report EPERM for a just-reaped process group instead of
        # ESRCH. Verify absence; never treat a live inaccessible group as gone.
        if sys.platform == "darwin":
            groups = subprocess.check_output(
                ["ps", "-A", "-o", "pgid=,stat="], text=True, timeout=2.0
            )
            members = [line.split() for line in groups.splitlines() if line.strip()]
            # poll() can race with the leader becoming a zombie. Zombies have
            # exited and cannot hold CUDA workers; wait() below reaps our child.
            if not any(
                len(row) >= 2 and row[0] == str(pgid) and not row[1].startswith("Z")
                for row in members
            ):
                return False
        raise


def _process_snapshot():
    if sys.platform.startswith("linux"):
        result = {}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                text = (path / "stat").read_text()
                # Fields following comm: state(3), ppid(4), ... starttime(22).
                fields = text[text.rfind(")") + 2 :].split()
                result[int(path.name)] = (int(fields[1]), fields[0], fields[19])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return result
    # POSIX ps is available on the supported Linux servers and macOS. Record
    # creation time as well as PID so an exited child's reused PID is not killed.
    output = subprocess.check_output(
        ["ps", "-e", "-o", "pid=,ppid=,stat=,lstart="], text=True, timeout=2.0
    )
    result = {}
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            result[int(fields[0])] = (int(fields[1]), fields[2], fields[3])
    return result


class OwnedProcessTree:
    """Remember descendants before parents exit, including separate sessions."""

    def __init__(self, root_pid, owner_tag=None):
        self.root_pid = root_pid
        self.owner_tag = owner_tag
        self.identities = {}
        snapshot = _process_snapshot()
        if root_pid in snapshot:
            self.identities[root_pid] = snapshot[root_pid][2]
        self._update(snapshot)

    def _update(self, snapshot):
        if self.owner_tag and sys.platform.startswith("linux"):
            marker = (self.owner_tag + "=1").encode()
            for pid, (_, state, born) in snapshot.items():
                if state.startswith("Z") or self.identities.get(pid) == born:
                    continue
                try:
                    environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                except OSError:
                    continue
                if marker in environment:
                    self.identities[pid] = born
        parents = {
            pid
            for pid, born in self.identities.items()
            if pid in snapshot and snapshot[pid][2] == born
        }
        while True:
            children = {
                pid
                for pid, (parent, _, _) in snapshot.items()
                if parent in parents and pid not in parents
            }
            if not children:
                break
            for pid in children:
                self.identities[pid] = snapshot[pid][2]
            parents.update(children)

    def live_descendants(self):
        snapshot = _process_snapshot()
        self._update(snapshot)
        return tuple(
            pid
            for pid, born in self.identities.items()
            if pid != self.root_pid
            and pid in snapshot
            and snapshot[pid][2] == born
            and not snapshot[pid][1].startswith("Z")
        )

    def signal_descendants(self, signum):
        for pid in self.live_descendants():
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    def wait_for_descendants(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = self.live_descendants()
            if not remaining:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"owned workers did not exit after SIGKILL: {remaining}"
                )
            time.sleep(0.05)


@uninterrupted_cleanup()
def stop_owned_process_group(process, *, timeout=5.0):
    """Only for children launched with start_new_session=True (PGID=PID)."""
    pgid = process.pid
    if pgid == os.getpgrp():
        raise RuntimeError("refusing to terminate the caller's process group")
    tree = OwnedProcessTree(pgid, getattr(process, "_think_bridge_owner_tag", None))
    if _group_alive(pgid) or tree.live_descendants():
        cleanup_message(
            f"[cleanup] stopping owned process group {pgid}; grace={timeout:g}s"
        )
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader; its exit alone does not prove worker exit.
            if not _group_alive(pgid) and not tree.live_descendants():
                break
            tree.live_descendants()  # Track late children while ancestors still exist.
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if _group_alive(pgid):
            cleanup_message(f"[cleanup] force-stopping owned process group {pgid}")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        tree.signal_descendants(signal.SIGKILL)
        tree.wait_for_descendants(timeout)
    process.wait(timeout=timeout)


@termination_signals()
def run_managed_process(command, *, env=None, shutdown_timeout_seconds=5.0):
    process = spawn_owned_process(command, env=env)
    try:
        return subprocess.CompletedProcess(command, process.wait())
    finally:
        stop_owned_process_group(process, timeout=shutdown_timeout_seconds)
