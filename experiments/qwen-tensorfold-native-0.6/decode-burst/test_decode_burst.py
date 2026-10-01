"""Exercise the installed native round; mock GPU boundaries, preserve native acceptance."""
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import multi

class BurstTests(unittest.TestCase):
    def decoder(self, burst=1):
        d = object.__new__(multi.MultiDecoder)
        d.w = object(); d.depth = 0; d.eos = (); d.converged = False
        d.decode_burst = burst; d._decode_since_fill = burst
        d.buf = SimpleNamespace(); d.gdn = object(); d.mbuf = None
        d.kept = []; d.free = []; d.held = {}; d.fills = {}; d.filling = []
        d.round_s = d.row_s = None; d.memory_gate = SimpleNamespace(waits=0, ends=0)
        st = SimpleNamespace(pos=1, mtp_len=1, capacity=10000, limit=10000)
        peer = Stream([1], 100, draft=False)
        peer.sid = 0; peer.st = st; peer.out = [1]; peer.context = [1]
        d.streams = {0: peer}
        return d

    def fill(self, d, sid=1, cancel=False):
        s = Stream([1] * 16, 4, draft=False)
        s.sid = sid; s.emit = lambda new: cancel
        d.filling.append(s); d.fills[sid] = []
        return s

    def gpu(self, events):
        stack = ExitStack()
        def stage(w, buf, windows):
            return [(st, i, i + len(tokens)) for i, (st, tokens) in enumerate(windows)]
        def compute(*args):
            events.append('decode'); return torch.tensor([[0., 0., 4.]])
        def commit(w, st, buf, rows, kept, **kwargs): st.pos += kept
        for obj, name, kw in [
            (multi, 'stage', {'side_effect': stage}), (multi, 'compute', {'side_effect': compute}),
            (multi, 'commit', {'side_effect': commit}),
            (multi.gdn_multi, 'Tables', {'return_value': object()}),
            (multi.gdn_multi, 'keep', {'return_value': [[]]}),
            (multi.attn_multi, 'Step', {'return_value': object()})]:
            stack.enter_context(patch.object(obj, name, **kw))
        return stack

    def test_default_order_matches_one_fill_then_decode_every_round(self):
        d = self.decoder(); self.fill(d); events = []
        with self.gpu(events), patch.object(d, '_pass', side_effect=lambda: events.append('fill') or []):
            for _ in range(5): d.round()
        self.assertEqual(events, ['fill', 'decode'] * 5)

    def test_four_actual_decodes_between_passes_and_bounded_progress(self):
        d = self.decoder(4); self.fill(d); events = []
        with self.gpu(events), patch.object(d, '_pass', side_effect=lambda: events.append('fill') or []):
            for _ in range(9): d.round()
        self.assertEqual(events, (['fill'] + ['decode'] * 4) * 2 + ['fill', 'decode'])
        self.assertEqual(d.streams[0].rounds, 9)

    def test_cancel_is_checked_even_when_fill_is_gated(self):
        d = self.decoder(4); s = self.fill(d, cancel=True); d._decode_since_fill = 1
        with self.gpu([]), patch.object(d, '_pass') as work:
            self.assertEqual(d.round(), [s]); work.assert_not_called()
        self.assertEqual(d.filling, []); self.assertEqual(d.fills, {})
        self.assertEqual(d.streams[0].rounds, 0)

    def test_decode_only_then_new_fill_gets_immediate_pass(self):
        d = self.decoder(4); d._decode_since_fill = 1; events = []
        with self.gpu(events), patch.object(d, '_pass', side_effect=lambda: events.append('fill') or []):
            d.round(); self.fill(d); d.round()
        self.assertEqual(events, ['decode', 'fill', 'decode'])

    def test_standalone_fill_is_not_gated(self):
        d = self.decoder(4); d.streams = {}; d._decode_since_fill = 1; s = self.fill(d)
        def complete():
            d.filling.remove(s); d.fills.pop(s.sid); return [s]
        with patch.object(d, '_pass', side_effect=complete) as work:
            self.assertEqual(d.round(), [s]); work.assert_called_once()

    def test_multiple_fills_keep_native_fifo_pass_ownership(self):
        d = self.decoder(4); a = self.fill(d); b = self.fill(d, 2); events = []; seen = []
        def step():
            s = d.filling.pop(0); d.fills.pop(s.sid); seen.append(s.sid)
            events.append('fill'); return [s]
        with self.gpu(events), patch.object(d, '_pass', side_effect=step):
            for _ in range(5): d.round()
        self.assertEqual(seen, [a.sid, b.sid])
        self.assertEqual(events, ['fill'] + ['decode'] * 4 + ['fill', 'decode'])

    def test_valid_environment_binds_burst_before_gpu_allocation(self):
        for value in ("1", "4"):
            d = object.__new__(multi.MultiDecoder)
            with patch.dict(os.environ, {"TF_FLASH_DECODE_BURST": value}):
                with self.assertRaisesRegex(ValueError, "one GPU"):
                    d.__init__(SimpleNamespace(comm=object()), slots=4, capacity=128)
            self.assertEqual(d.decode_burst, int(value))
            self.assertEqual(d._decode_since_fill, int(value))

    def test_invalid_environment_fails_before_allocating_gpu(self):
        for value in ('0', '2', '4.0', '-1', 'garbage', ''):
            with patch.dict(os.environ, {'TF_FLASH_DECODE_BURST': value}):
                with self.assertRaisesRegex(ValueError, 'must be 1 or 4'):
                    multi.MultiDecoder(object(), slots=4, capacity=128)

if __name__ == '__main__': unittest.main()
