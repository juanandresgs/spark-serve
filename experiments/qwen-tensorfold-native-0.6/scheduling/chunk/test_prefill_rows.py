"""Native constructor allocation sizes and actual FIFO packing; mock GPU allocators."""
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from tensorfold.families.qwen4_exp.cuda import multi

class PrefillRowsTests(unittest.TestCase):
    def construct(self, env, argument=2048, converged=False):
        w = SimpleNamespace(comm=None, cfg=SimpleNamespace(eos=()), mtp=None, draft_ids=None)
        allocations = []
        def buffers(w, rows, capacity, **kwargs):
            allocations.append((rows, kwargs)); return SimpleNamespace()
        state = SimpleNamespace(cache_bytes=lambda capacity: 0)
        with ExitStack() as stack:
            clean = dict(os.environ); clean.pop('TF_FLASH_PREFILL_ROWS', None)
            clean['TF_FLASH_DECODE_BURST'] = '4'
            if env is not None: clean['TF_FLASH_PREFILL_ROWS'] = env
            stack.enter_context(patch.dict(os.environ, clean, clear=True))
            stack.enter_context(patch.object(multi, 'Buffers', side_effect=buffers))
            stack.enter_context(patch.object(multi, 'State', return_value=state))
            stack.enter_context(patch.object(multi, '_tensors', return_value=[]))
            stack.enter_context(patch.object(multi, 'converges', return_value=converged))
            stack.enter_context(patch.object(multi.gdn_multi, 'Scratch', return_value=object()))
            stack.enter_context(patch.object(multi.torch.cuda, 'is_available', return_value=False))
            d = multi.MultiDecoder(w, slots=4, capacity=8192, depth=6, prefill_rows=argument)
        return d, allocations

    def test_valid_values_bind_property_and_real_buffer_constructor_argument(self):
        for value in (1024, 2048):
            d, allocations = self.construct(str(value))
            self.assertEqual(d.prefill_rows, value)
            self.assertEqual(allocations, [(28, {'moe_prefill': True}), (value, {'prefill': True})])
            self.assertEqual(d.decode_burst, 4)

    def test_unset_preserves_supplied_constructor_value(self):
        d, allocations = self.construct(None, argument=1536)
        self.assertEqual(d.prefill_rows, 1536)
        self.assertEqual(allocations[-1], (1536, {'prefill': True}))

    def test_converged_buffer_keeps_decode_window_addition(self):
        d, allocations = self.construct('1024', converged=True)
        self.assertEqual(d.prefill_rows, 1024)
        self.assertEqual(allocations[-1], (1052, {'prefill': True}))

    def test_invalid_values_fail_before_gpu_allocation(self):
        for value in ('', '0', '512', '4096', '1024.0', 'garbage'):
            with self.assertRaisesRegex(ValueError, 'must be 1024 or 2048'):
                self.construct(value)

    def test_actual_pieces_obeys_both_bounds_and_partial_fifo(self):
        for bound in (1024, 2048):
            d, _ = self.construct(str(bound))
            a = SimpleNamespace(sid=1, prompt=[1] * 800, background=False)
            b = SimpleNamespace(sid=2, prompt=[1] * 4096, background=False)
            d.filling = [a, b]; d.fills = {1: [None, False, 100, None], 2: [None, False, 50, None]}
            pieces = d._pieces()
            self.assertEqual([(s.sid, start, n) for s, start, n in pieces],
                             [(1, 100, 700), (2, 50, bound - 700)])
            self.assertEqual(sum(n for _, _, n in pieces), bound)
            # Packing plans do not mutate prompt offsets or the retained snapshot slot.
            self.assertEqual(d.fills[1], [None, False, 100, None])
            self.assertEqual(d.fills[2], [None, False, 50, None])

if __name__ == '__main__': unittest.main()
