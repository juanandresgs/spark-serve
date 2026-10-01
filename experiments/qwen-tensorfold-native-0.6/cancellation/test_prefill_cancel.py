"""Actual native scheduler + prefill loop; mock only GPU work and state buffers."""
import socket
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation


class FakeGpuDecoder(MultiDecoder):
    def __init__(self):
        self.filling, self.fills, self.streams = [], {}, {}
        self.kept, self.free, self.held = [], [], {}
        self.next_id = 0
        self.admitted = threading.Event()
        self.release = threading.Event()
        self.passes = 0
        self.tokens = [7]
        self.finished = []
        self.survivor_steps = 0
    def admit(self, stream):
        stream.sid = self.next_id
        self.next_id += 1
        stream.st = SimpleNamespace()
        self.filling.append(stream)
        self.fills[stream.sid] = []
        self.admitted.set()
    def _pass(self):
        time.sleep(.01)  # one bounded GPU pass, no model/GPU imports beyond the real module
        self.passes += 1
        if not self.release.is_set():
            return []
        done = list(self.filling)
        for stream in done:
            self.filling.remove(stream)
            self.fills.pop(stream.sid)
            stream.take(self.tokens)
            stream.done = True
        return done
    def round(self):
        # Real prefill-loop cancellation executes; one already-decoding peer keeps progressing.
        done = self._fill()
        if self.streams:
            self.survivor_steps += 1
        return done
    def finish(self, done):
        for stream in done:
            self.finished.append(stream)
            self.streams.pop(stream.sid, None)
            self.free.append(stream.st)
    def drop(self):
        done = list(self.filling)
        self.filling, self.fills = [], {}
        return done


def request(scheduler, connection, result, key):
    cancel = socket_cancellation(connection)
    tokens = []
    def emit(new):
        tokens.extend(new)
        return cancel.cancelled
    try:
        stats = scheduler.submit([1] * 128, 10, None, True, emit)
        result[key] = ('done', stats, tokens)
    except RequestCancelled:
        result[key] = ('cancelled', tokens)
    except Exception as exc:
        result[key] = ('error', type(exc).__name__)


class CancellationTests(unittest.TestCase):
    def test_socket_close_during_prefill_reclaims_before_first_token(self):
        decoder = FakeGpuDecoder(); scheduler = Scheduler(decoder)
        client, server = socket.socketpair(); result = {}
        thread = threading.Thread(target=request, args=(scheduler, server, result, 'one'), daemon=True)
        thread.start(); self.assertTrue(decoder.admitted.wait(1))
        client.close(); thread.join(.75)
        try:
            self.assertFalse(thread.is_alive(), 'cancel waits for first token instead of checking prefill')
            self.assertEqual(result['one'], ('cancelled', []))
            self.assertEqual(decoder.live(), 0)
            self.assertEqual(scheduler.boxes, {})
            self.assertTrue(scheduler.waiting.empty())
            self.assertEqual(len(decoder.free), 1)
        finally:
            decoder.release.set(); thread.join(1); server.close()

    def test_cancelled_prefill_does_not_stop_peer_and_queued_client_is_not_admitted(self):
        decoder = FakeGpuDecoder(); scheduler = Scheduler(decoder, max_streams=1)
        client, server = socket.socketpair(); result = {}
        first = threading.Thread(target=request, args=(scheduler, server, result, 'first'), daemon=True)
        first.start(); self.assertTrue(decoder.admitted.wait(1))
        # A real decoder has an active decode stream; its progress is represented at each round boundary.
        decoder.streams[99] = SimpleNamespace(done=False, waiting=False, background=False)
        client2, server2 = socket.socketpair()
        second = threading.Thread(target=request, args=(scheduler, server2, result, 'queued'), daemon=True)
        second.start(); client2.close()
        time.sleep(.15); before = decoder.survivor_steps; client.close(); first.join(.75)
        try:
            self.assertFalse(first.is_alive())
            self.assertEqual(result['first'], ('cancelled', []))
            self.assertGreater(decoder.survivor_steps, before)
            decoder.streams.clear()  # peer finishes; queued cancellation must not allocate a slot
            second.join(.75)
            self.assertFalse(second.is_alive())
            self.assertEqual(result['queued'], ('cancelled', []))
            self.assertEqual(decoder.next_id, 1)
            self.assertEqual(scheduler.boxes, {})
            self.assertTrue(scheduler.waiting.empty())
        finally:
            decoder.streams.clear(); decoder.release.set()
            first.join(1); second.join(1); server.close(); server2.close()



class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return ''.join({7: 'OK', 8: '<eos>'}.get(i, 'p') for i in ids)
    def token_to_id(self, value):
        return 9 if value == '</think>' else None


class AppEngine:
    concurrent = True
    eos = (8,)
    def __init__(self, decoder):
        self.scheduler = Scheduler(decoder)
    def generate(self, prompt, count, sampling, emit, stop_eos=True, **options):
        return self.scheduler.submit(prompt, count, sampling, options.get('draft', True), emit, stop_eos)


def native_app(decoder):
    app = object.__new__(App)
    app.engine = AppEngine(decoder)
    app.tok = ToyTokenizer()
    return app


class ProductionCallbackTests(unittest.TestCase):
    def test_healthy_nonstream_success_callback_does_not_cancel_empty_heartbeat(self):
        decoder = FakeGpuDecoder(); decoder.tokens = [7, 8]; decoder.release.set()
        app = native_app(decoder)
        result = app.run({'messages': [{'role': 'user', 'content': 'hi'}]}, True, lambda delta: True,
                         prepared=PreparedRequest([1], 10, [], False, None), cancelled=lambda: False)
        self.assertEqual(result['content'], 'OK')
        self.assertEqual(result['finish'], 'stop')
        self.assertEqual(result['completion_tokens'], 2)
        self.assertEqual(app.engine.scheduler.boxes, {})

    def test_healthy_streaming_success_callback_emits_no_empty_sse_delta(self):
        decoder = FakeGpuDecoder(); decoder.tokens = [7, 8]; decoder.release.set()
        app = native_app(decoder); emitted = []
        def emit(delta):
            emitted.append(delta)
            return True  # exact successful HTTP stream callback convention
        result = app.run({'messages': [{'role': 'user', 'content': 'hi'}]}, True, emit,
                         prepared=PreparedRequest([1], 10, [], False, None), cancelled=lambda: False)
        self.assertEqual(result['content'], 'OK')
        self.assertEqual(emitted, [{'content': 'OK'}])
        self.assertEqual(result['completion_tokens'], 2)

    def test_real_app_wrapper_disconnect_before_token_cancels_and_emits_nothing(self):
        decoder = FakeGpuDecoder(); app = native_app(decoder)
        client, server = socket.socketpair(); cancellation = socket_cancellation(server)
        result, emitted = {}, []
        def invoke():
            try:
                app.run({'messages': [{'role': 'user', 'content': 'hi'}]}, True,
                        lambda delta: (emitted.append(delta), True)[1],
                        prepared=PreparedRequest([1] * 128, 10, [], False, None),
                        cancelled=lambda: cancellation.cancelled)
                result['status'] = 'unexpected_success'
            except RequestCancelled:
                result['status'] = 'cancelled'
        thread = threading.Thread(target=invoke, daemon=True)
        thread.start(); self.assertTrue(decoder.admitted.wait(1)); client.close(); thread.join(.75)
        try:
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, {'status': 'cancelled'})
            self.assertEqual(emitted, [])
            self.assertEqual(app.engine.scheduler.boxes, {})
            self.assertEqual(decoder.live(), 0)
        finally:
            decoder.release.set(); thread.join(1); server.close()



class ActualRoundTests(unittest.TestCase):
    def test_actual_round_yields_cancelled_fill_then_peer_progresses_and_cleans_up(self):
        import torch
        from tensorfold.cuda.streams import Stream
        from tensorfold.families.qwen4_exp.cuda import multi
        if not hasattr(multi.MultiDecoder, '_cancel_filling'):
            self.skipTest('The targeted early-return branch exists only in the patch')
        decoder = object.__new__(multi.MultiDecoder)
        decoder.w = object()
        decoder.depth, decoder.eos, decoder.converged = 0, (), True
        decoder.buf, decoder.gdn, decoder.mbuf = SimpleNamespace(), object(), None
        decoder.kept, decoder.free, decoder.held = [], [], {}
        decoder.memory_gate = SimpleNamespace(waits=0, ends=0)
        decoder.round_s = decoder.row_s = None
        resets = []
        def state(name):
            return SimpleNamespace(pos=1, mtp_len=1, capacity=64, limit=64,
                                   reset=lambda w: resets.append(name))
        cancelled = Stream([1] * 16, 4, draft=False)
        cancelled.sid, cancelled.st, cancelled.emit = 1, state('cancelled'), lambda new: True
        peer = Stream([1], 2, draft=False)
        peer.sid, peer.st, peer.out, peer.context = 2, state('peer'), [1], [1]
        decoder.filling, decoder.fills = [cancelled], {1: []}
        decoder.streams = {2: peer}
        # Mock only GPU staging/forward/recurrent-buffer functions. Keep native round,
        # admission-memory check, target sampler, acceptance, Stream.take and finish.
        def stage(w, buf, windows):
            return [(st, i, i + len(tokens)) for i, (st, tokens) in enumerate(windows)]
        def commit(w, st, buf, rows, kept, **kwargs):
            st.pos += kept
        with patch.object(multi, 'stage', side_effect=stage), \
             patch.object(multi, 'compute', return_value=torch.tensor([[0., 0., 4.]])) as compute, \
             patch.object(multi, 'commit', side_effect=commit), \
             patch.object(multi.gdn_multi, 'Tables', return_value=object()), \
             patch.object(multi.gdn_multi, 'keep', return_value=[[]]), \
             patch.object(multi.attn_multi, 'Step', return_value=object()):
            first = decoder.round()
            self.assertEqual(first, [cancelled])
            compute.assert_not_called()
            self.assertEqual(peer.out, [1])
            decoder.finish(first)
            self.assertEqual(decoder.filling, [])
            self.assertEqual(decoder.fills, {})
            self.assertEqual(resets, ['cancelled'])
            second = decoder.round()
            self.assertEqual(second, [peer])
            self.assertEqual(peer.out, [1, 2])
            self.assertEqual(peer.rounds, 1)
            compute.assert_called_once()
            decoder.finish(second)
        self.assertEqual(decoder.streams, {})
        self.assertEqual(decoder.held, {})
        self.assertEqual(resets, ['cancelled', 'peer'])
        self.assertEqual(len(decoder.free), 2)

if __name__ == '__main__':
    unittest.main()
