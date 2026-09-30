"""Site-owned cooperative scheduler over the pinned runtime's unmodified math.

One dispatcher owns execution. Admission workers preserve the original prefill
call stack, park after committed chunks, and resume only when granted a turn.
A CUDA synchronization before every handoff prevents cross-thread scratch use.
No forward pass, sampling implementation, or model state arithmetic is copied.
"""
from collections import deque
import os
import queue
import select
import socket
import threading
import time

_current = threading.local()
_http = threading.local()

class ClientGone(Exception):
    spark_client_cancelled = True

def peer_gone(sock):
    if sock is None:
        return False
    try:
        if not select.select([sock], [], [], 0)[0]:
            return False
        return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
    except BlockingIOError:
        return False
    except (OSError, ValueError):
        return True


class Admission:
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream
        self.go = threading.Event()
        self.parked = threading.Event()
        self.finished = False
        self.error = None
        self.chunks = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def checkpoint(self):
        self.chunks += 1
        if self.stream._spark_cancel.is_set():
            raise ClientGone('Client disconnected during prefill')
        # The last chunk must finish first-token sampling without another wait.
        if self.state.pos >= len(self.stream.prompt):
            return
        self.owner.synchronize()
        self.parked.set()
        self.go.wait()
        self.go.clear()
        if self.stream._spark_cancel.is_set():
            raise ClientGone('Client disconnected while prefill was parked')

    def run(self):
        self.go.wait()
        self.go.clear()
        _current.admission = self
        try:
            if self.stream._spark_cancel.is_set():
                raise ClientGone('Client disconnected before admission')
            self.owner.decoder.admit(self.stream)
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.owner.synchronize()
            except BaseException as exc:
                self.error = exc
            self.finished = True
            self.parked.set()
            _current.admission = None

    def step(self):
        self.parked.clear()
        self.go.set()
        # Do not dispatch anything else if GPU work hangs: shared scratch is owned.
        self.parked.wait()

class CooperativeScheduler:
    def __init__(self, decoder, *, max_streams=4, synchronize=None):
        self.decoder = decoder
        self.max_streams = max_streams
        if synchronize is None:
            import torch
            synchronize = torch.cuda.synchronize
        self.synchronize = synchronize
        self.incoming = queue.Queue()
        self.pending = deque()
        self.jobs = deque()
        self.replies = {}
        self.short_tokens = int(os.environ.get('SPARK_SHORT_PROMPT_TOKENS', '2048'))
        self.decode_burst = int(os.environ.get('SPARK_DECODE_BURST', '1'))
        assert 1 <= self.decode_burst <= 8
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, prompt, count, sampling, draft, emit):
        from tensorfold.cuda.streams import Stream
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft)
        reply = queue.Queue()
        stopped = threading.Event()
        stream._spark_cancel = stopped
        connection = getattr(_http, 'connection', None)
        def output(tokens):
            reply.put(('tokens', tokens))
            return stopped.is_set()
        stream.emit = output
        self.incoming.put((stream, reply, time.monotonic()))
        while True:
            if peer_gone(connection):
                stopped.set()
            try:
                kind, payload = reply.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == 'tokens':
                if not stopped.is_set() and emit(payload):
                    stopped.set()
            elif kind == 'error':
                raise payload
            else:
                return payload

    def complete(self, streams):
        self.decoder.finish(streams)
        for stream in streams:
            reply = self.replies.pop(id(stream), None)
            if reply is not None:
                reply.put(('done', stream.stats()))

    def accept(self):
        while True:
            try:
                self.pending.append(self.incoming.get_nowait())
            except queue.Empty:
                break
        while self.pending and len(self.jobs) + self.decoder.live() < self.max_streams:
            # Admission is cheap; actual work is scheduled one committed chunk at a time.
            short = next((x for x in self.pending if len(x[0].prompt) <= self.short_tokens), None)
            aged = self.pending[0] if time.monotonic() - self.pending[0][2] >= 10 else None
            item = aged if aged is not None else (short if short is not None else self.pending[0])
            self.pending.remove(item)
            stream, reply, _ = item
            self.replies[id(stream)] = reply
            job = Admission(self, stream)
            if len(stream.prompt) <= self.short_tokens:
                self.jobs.appendleft(job)
            else:
                self.jobs.append(job)

    def run(self):
        while True:
            if not self.jobs and not self.decoder.live() and not self.pending:
                self.pending.append(self.incoming.get())
            self.accept()
            if self.decoder.live():
                try:
                    for _ in range(self.decode_burst):
                        done = self.decoder.round()
                        self.synchronize()
                        self.complete(done)
                        if not self.decoder.live():
                            break
                except Exception as exc:
                    for stream in self.decoder.drop():
                        reply = self.replies.pop(id(stream), None)
                        if reply is not None:
                            reply.put(('error', exc))
            if self.jobs:
                job = self.jobs.popleft()
                job.step()
                if not job.finished:
                    self.jobs.append(job)
                elif job.error is not None:
                    reply = self.replies.pop(id(job.stream))
                    reply.put(('error', job.error))
                elif job.stream.done:
                    self.complete([job.stream])


def install():
    import tensorfold.cuda.scheduler as scheduling
    from tensorfold.families.qwen4_exp.cuda import decode, multi
    original_commit = decode.commit
    original_slot = multi.MultiDecoder._slot_for

    def reserve(decoder, prompt, reuse):
        result = original_slot(decoder, prompt, reuse)
        job = getattr(_current, 'admission', None)
        if job is not None:
            job.state = result[0]
        return result

    def committed(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        job = getattr(_current, 'admission', None)
        if job is not None:
            job.checkpoint()
        return result

    multi.MultiDecoder._slot_for = reserve
    decode.commit = committed
    scheduling.Scheduler = CooperativeScheduler
    import tensorfold.cuda.server as server
    previous_handler = server.make_handler
    def handler(app):
        parent = previous_handler(app)
        class DisconnectAware(parent):
            def do_POST(self):
                _http.connection = self.connection
                try:
                    return super().do_POST()
                finally:
                    _http.connection = None
        return DisconnectAware
    server.make_handler = handler
