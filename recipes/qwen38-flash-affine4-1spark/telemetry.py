"""Site-owned admission counters and Prometheus compatibility for sparkDash.

Admission permits match the actual configured engine streams. Decode tokens count committed
callback tokens; prompt totals count completed requests (including cache hits).
No fabricated KV utilization or latency percentiles.
"""
import threading
from functools import wraps


def install():
    import tensorfold.cuda.server as server
    original_init = server.App.__init__
    original_handler = server.make_handler

    def initialize(app, *args, **kwargs):
        original_init(app, *args, **kwargs)
        lock = threading.Lock()
        slots = int(app.engine.scheduler.max_streams)
        assert slots in (4, 5), 'Unqualified stream allocation'
        permits = threading.BoundedSemaphore(slots)
        counts = dict(running=0, waiting=0, prompt=0, output=0, completed=0, errors=0, cancelled=0)
        app.spark_counts, app.spark_lock = counts, lock
        generate = app.engine.generate

        @wraps(generate)
        def measured(prompt, max_tokens, sampling, on_tokens, **options):
            with lock: counts['waiting'] += 1
            permits.acquire()
            with lock:
                counts['waiting'] -= 1
                counts['running'] += 1
            def tokens(new):
                with lock: counts['output'] += len(new)
                return on_tokens(new)
            try:
                result = generate(prompt, max_tokens, sampling, tokens, **options)
                with lock:
                    counts['prompt'] += len(prompt)
                    counts['completed'] += 1
                return result
            except BaseException as exc:
                with lock: counts['cancelled' if getattr(exc, 'spark_client_cancelled', False) else 'errors'] += 1
                raise
            finally:
                with lock: counts['running'] -= 1
                permits.release()
        app.engine.generate = measured

    def handler(app):
        parent = original_handler(app)
        class ObservedHandler(parent):
            def do_GET(self):
                path = self.path.rstrip('/')
                if path == '/metrics':
                    with app.spark_lock: c = dict(app.spark_counts)
                    names = {'vllm:num_requests_running':'running','vllm:num_requests_waiting':'waiting',
                             'vllm:prompt_tokens_total':'prompt','vllm:generation_tokens_total':'output',
                             'spark_tensorfold_requests_completed_total':'completed','spark_tensorfold_errors_total':'errors',
                             'spark_tensorfold_requests_cancelled_total':'cancelled'}
                    text = '# TensorFold compatibility metrics; prompt totals include cached input.\n'
                    text += ''.join(f'{name} {c[key]}\n' for name,key in names.items())
                    data = text.encode()
                    self.send_response(200)
                    self.send_header('Content-Type','text/plain; version=0.0.4')
                    self.send_header('Content-Length',str(len(data)))
                    self.end_headers();self.wfile.write(data)
                elif path in ('/v1/models','/models'):
                    self._json(200,{'object':'list','data':[{'id':app.served,'object':'model','owned_by':'tensorfold',
                        'max_model_len':262144,'meta':{'runtime':'TensorFold 0.3.6.1 + site tool adapter','parallel':4}}]})
                else:
                    super().do_GET()
        return ObservedHandler
    server.App.__init__ = initialize
    server.make_handler = handler
