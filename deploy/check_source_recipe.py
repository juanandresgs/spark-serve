"""Exercise the standalone catalog contract without a GPU or model download."""
import copy
from pathlib import Path
from spark_serve.config import ConfigError
from spark_serve.recipes import read_recipe, check, prepare

root = Path(__file__).resolve().parents[1]
recipe = read_recipe(root, 'qwen38-flash-affine4-1spark')
assert check(root, recipe)['valid']
wrong = copy.deepcopy(recipe)
wrong['source_pins']['target_revision'] = '0' * 40
assert not check(root, wrong)['valid'], 'model pin drift must fail'
wrong = copy.deepcopy(recipe)
del wrong['assets'][wrong['pins_file']]
assert not check(root, wrong)['valid'], 'unhashed pins must fail'
wrong = copy.deepcopy(recipe)
wrong['portable_adapter'] = True
assert not check(root, wrong)['valid'], 'unsupported broker claim must fail'
try:
    prepare(root, recipe, root / 'nonexistent-site.json', root / 'must-not-be-created')
except ConfigError as exc:
    assert recipe['guide'] in str(exc)
else:
    raise AssertionError('standalone recipe cannot prepare broker configuration')
print('Standalone source contract: 5 checks passed')
