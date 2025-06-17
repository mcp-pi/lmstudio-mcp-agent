import subprocess

def init_dataset():
    subprocess.run("python3 utils/download.py", shell=True, check=True)
    subprocess.run("python3 utils/make.py", shell=True, check=True)
    subprocess.run("python3 utils/remove.py", shell=True, check=True)
    subprocess.run("echo 'OK'", shell=True, check=True)
