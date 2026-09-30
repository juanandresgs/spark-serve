"""Verify positional reads with real files, duplicates, boundaries and EOF."""
import concurrent.futures,os,tempfile,types
import numpy as np
from ssd_pipeline import native_gather
from spark_positional_reader import read_rows
rng=np.random.default_rng(147)
with tempfile.TemporaryFile() as f:
    words=rng.integers(0,2**32,size=(257,8),dtype=np.uint32)
    scales=rng.integers(0,65536,size=(257,2),dtype=np.uint16)
    biases=rng.integers(0,65536,size=(257,2),dtype=np.uint16)
    f.write(words.tobytes()+scales.tobytes()+biases.tobytes());f.flush()
    table=types.SimpleNamespace(_closer=types.SimpleNamespace(alive=True),rows=257,starts=np.array([0,257]),fidx=np.array([0]),_fd_of=np.array([f.fileno()]),bases=np.array([[0,words.nbytes,words.nbytes+scales.nbytes]]),wrow=32,grow=4)
    for ids in [np.array([],dtype=np.int64),np.array([0,256,1,0]),rng.integers(0,257,size=(2048,16))]:
        got=native_gather(table,ids)
        for actual,expected in zip(got,(words,scales,biases)):
            np.testing.assert_array_equal(actual,expected[ids.reshape(-1)])
    for ids in [np.array([-1]),np.array([257]),np.array([1.5])]:
        try:native_gather(table,ids)
        except ValueError:pass
        else:raise AssertionError('invalid row admitted')
    try:read_rows(np.array([f.fileno()]),np.array([10**9]),32,2)
    except RuntimeError:pass
    else:raise AssertionError('EOF accepted')
    # Overlap independent calls, including read failures, on the same process pool.
    def concurrent_read(i):
        if i % 9 == 0:
            try:read_rows(np.array([f.fileno()]),np.array([10**9]),32,8)
            except RuntimeError:return True
            raise AssertionError('Concurrent EOF accepted')
        ids=np.random.default_rng(i).integers(0,257,size=(1 if i%2 else 32768),dtype=np.int64)
        got=native_gather(table,ids)
        for actual,expected in zip(got,(words,scales,biases)):
            np.testing.assert_array_equal(actual,expected[ids])
        return True
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        assert all(pool.map(concurrent_read,range(128)))
print('native reader real-file correctness passed')
