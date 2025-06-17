import subprocess

subprocess.run("rm data/*.csv", shell=True, check=True)
subprocess.run("rm data/*.parquet", shell=True, check=True)
