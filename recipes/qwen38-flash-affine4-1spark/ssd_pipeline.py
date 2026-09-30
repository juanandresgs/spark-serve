"""Independent I/O adapters over pinned TensorFold interfaces.

Gather preserves the upstream byte format and row order. Read-ahead holds at
most the current and next prefill chunk per admission, scoped to that request.
It invokes the dependency's hashing; no hashing or inference math is copied.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import numpy as np

_local = threading.local()
_pool = ThreadPoolExecutor(2, thread_name_prefix='spark-read-ahead')

def native_gather(table, ids):
    from spark_positional_reader import read_rows
    if not table._closer.alive:
        raise ValueError('closed table')
    x = np.asarray(ids).reshape(-1)
    if x.size and (x.dtype.kind not in 'iu' or x.min() < 0 or x.max() >= table.rows):
        raise ValueError('invalid table row')
    unique, restore = np.unique(x.astype(np.int64), return_inverse=True)
    shard = np.searchsorted(table.starts, unique, side='right') - 1
    relative = unique - table.starts[shard]
    fds = table._fd_of[table.fidx[shard]]
    parts = []
    for component, width in enumerate((table.wrow, table.grow, table.grow)):
        offsets = table.bases[shard, component] + relative * width
        raw = read_rows(fds, offsets, width, int(os.environ.get('SPARK_IO_THREADS', '8')))
        parts.append(raw[restore].view(np.uint32 if component == 0 else np.uint16))
    return tuple(parts)

def key(table, ids):
    return id(table), np.asarray(ids, dtype=np.int64).tobytes()

def install():
    from tensorfold.families.qwen4_exp.ssd_table import SSDTable
    from tensorfold.families.qwen4_exp.cuda import decode, multi
    prior_gather = SSDTable.gather
    read = native_gather if os.environ.get('SPARK_NATIVE_IO', '1') == '1' else prior_gather
    def gather(table, ids):
        plan = getattr(_local, 'plan', None)
        future = None if plan is None else plan['ready'].pop(key(table, ids), None)
        return future.result() if future is not None else read(table, ids)
    SSDTable.gather = gather
    if os.environ.get('SPARK_READ_AHEAD', '1') != '1':
        return
    prior_prefill, prior_forward = multi.prefill, decode.forward
    def prefill(engine, prompt, *args, **kwargs):
        previous = getattr(_local, 'plan', None)
        plan = dict(prompt=prompt, rows=engine.prefill_rows, ready={})
        _local.plan = plan
        try:
            return prior_prefill(engine, prompt, *args, **kwargs)
        finally:
            for future in plan['ready'].values():
                future.cancel()
            # Readers own references to their arrays/table until they finish.
            _local.plan = previous
    def forward(weights, state, buffers, tokens, **kwargs):
        plan = getattr(_local, 'plan', None)
        if plan is not None:
            end = state.pos + len(tokens)
            following = plan['prompt'][end:end+plan['rows']]
            for layer in weights.layers:
                ple = layer.ple
                if ple is None or not isinstance(ple.table, SSDTable):
                    continue
                current = ple.ngram.ids(state.ple_history, np.asarray(tokens, dtype=np.int64))
                todo = [current]
                if following:
                    history = np.concatenate((state.ple_history, np.asarray(tokens, dtype=np.int64)))[-ple.ngram.context:]
                    todo.append(ple.ngram.ids(history, np.asarray(following, dtype=np.int64)))
                for ids in todo:
                    k = key(ple.table, ids)
                    if k not in plan['ready']:
                        plan['ready'][k] = _pool.submit(read, ple.table, ids)
        return prior_forward(weights, state, buffers, tokens, **kwargs)
    multi.prefill = prefill
    decode.forward = forward
