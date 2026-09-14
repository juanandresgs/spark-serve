import os,subprocess,sys
env=dict(os.environ,EXL3_FAT_GROUPED="0")
subprocess.run([sys.executable,"/opt/glm53/test_exl3_e2_regression.py"],env=env,check=True)
if os.environ.get("EXL3_SELFCHECK_GPU","1")!="0":
 subprocess.run([sys.executable,"/opt/glm53/e3-trial-check.py","--small","--out","/tmp/e3-trial-check.json"],check=True)
