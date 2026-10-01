"""Portable speed fixture schedule; warmups never enter measured groups."""
import hashlib,json
from transport import FIXTURES

VERSION='cell-warmup-v2'
CELLS=((1,'code'),(1,'prose'),(4,'code'),(4,'prose'))

def measured_cells(rep):return list(CELLS if rep%2==0 else reversed(CELLS))
def measured_number(rep,index):return 50000+rep*10000+512+index
def warmup_cells():return [(c,kind,[900000+cell*100+i for i in range(4)]) for cell,(c,kind) in enumerate(CELLS)]
def metadata(version):
    return {'version':version,'cap':512,'requests_per_cell':4,'measured_repetitions_primary':4,'cell_order_even':list(CELLS),'cell_order_odd':list(reversed(CELLS)),'exact_warmup':{'fixture':99001,'cap':128,'excluded':True},'representative_warmup_cells':warmup_cells() if version==VERSION else [],'warmups_excluded':True,'fixtures_sha256':hashlib.sha256(json.dumps(FIXTURES,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'measured_number_formula':'50000 + rep*10000 + 512 + index; same IDs for C1/C4; kind distinguishes code/prose','sampling':{'temperature':0,'seed':'1000 + fixture_number','thinking':False,'reasoning_effort':'medium in top-level and chat_template_kwargs'}}
