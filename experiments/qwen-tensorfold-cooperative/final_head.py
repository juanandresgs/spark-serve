"""Skip the unused vocabulary projection of intermediate prompt chunks.

The original forward still performs all state-producing work. Its public logits
switch avoids the head, and an empty view satisfies the original prefill loop's
unused intermediate clone. The last chunk uses the ordinary full forward.
"""
import threading
_context=threading.local()

def install():
    from tensorfold.families.qwen4_exp.cuda import decode,multi
    original_prefill=multi.prefill
    original_forward=decode.forward
    def prefill(engine,prompt,*args,**kwargs):
        previous=getattr(_context,'length',None)
        _context.length=len(prompt)
        try:return original_prefill(engine,prompt,*args,**kwargs)
        finally:_context.length=previous
    def forward(weights,state,buffers,tokens,**kwargs):
        length=getattr(_context,'length',None)
        if length is not None and state.pos+len(tokens)<length:
            original_forward(weights,state,buffers,tokens,logits=False)
            return buffers.logits[:0]
        return original_forward(weights,state,buffers,tokens,**kwargs)
    multi.prefill=prefill
    decode.forward=forward
