"""Site-owned startup; only pinned upstream TensorFold is third-party code."""
import os
from tensorfold.families.qwen4_exp.cuda import multi, decode, exl3
from tensorfold.cuda import geometry
rows = int(os.environ.get('SPARK_PREFILL_ROWS', '2048'))
assert rows in (2048, 4096)
multi.PREFILL_ROWS = decode.PREFILL_ROWS = exl3.PREFILL_ROWS = geometry.PREFILL_ROWS = rows
if os.environ.get('SPARK_SSD_PIPELINE', '1') == '1':
    from ssd_pipeline import install
    install()
if os.environ.get('SPARK_TILED_SELECT', '0') == '1':
    from selection import install
    install()
if os.environ.get('SPARK_COOPERATIVE', '1') == '1':
    from cooperative import install
    install()
from tool_adapter import install
install()
from telemetry import install
install()
from reliability import install
install()
from tensorfold.cli import main
main()
