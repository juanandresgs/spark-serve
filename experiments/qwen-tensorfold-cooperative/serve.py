"""Load only site-authored adapters over an immutable runtime image."""
import os
from tensorfold.families.qwen4_exp.cuda import multi,decode,exl3
from tensorfold.cuda import geometry
rows = int(os.environ.get('SPARK_PREFILL_ROWS', '2048'))
assert rows in (512, 1024, 2048, 4096, 8192)
multi.PREFILL_ROWS = decode.PREFILL_ROWS = exl3.PREFILL_ROWS = geometry.PREFILL_ROWS = rows
if os.environ.get('SPARK_COOPERATIVE', '1') == '1':
    from cooperative import install
    install()
if os.environ.get('SPARK_FINAL_HEAD', '0') == '1':
    from final_head import install
    install()
if os.environ.get('SPARK_PROFILE', '0') == '1':
    from profile_blocks import install
    install()
from tool_adapter import install
install()
from telemetry import install
install()
from tensorfold.cli import main
main()
