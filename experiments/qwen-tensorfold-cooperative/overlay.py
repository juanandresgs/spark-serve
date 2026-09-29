#!/usr/bin/env python3
"""Honor checkpoint-published generation EOS settings without modifying weights."""
import hashlib,json,os
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
root=a.directory.resolve();target=root/'target'
config=json.loads((target/'config.json').read_text());generation=json.loads((target/'generation_config.json').read_text())
ids=generation['eos_token_id'];assert ids==[248046,248044]
tokens=json.loads((target/'tokenizer.json').read_text())['added_tokens']
assert {t['content']:t['id'] for t in tokens if t['content'] in ['<|im_end|>','<|endoftext|>']}=={'<|im_end|>':248046,'<|endoftext|>':248044}
# Root EOS is the loader's stopping contract. Preserve text_config.eos_token_id,
# which also has a separate n-gram/PLE role in the model geometry.
config['eos_token_id']=ids
out=root/'config-overlay.json';next_path=out.with_suffix('.next');next_path.write_text(json.dumps(config,indent=2)+'\n');os.replace(next_path,out)
receipt={'change':'root eos_token_id from checkpoint generation_config','eos_token_id':ids,'original_config_sha256':hashlib.sha256((target/'config.json').read_bytes()).hexdigest(),'generation_config_sha256':hashlib.sha256((target/'generation_config.json').read_bytes()).hexdigest(),'overlay_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(root/'overlay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
