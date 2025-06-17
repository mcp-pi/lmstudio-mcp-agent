import subprocess

subprocess.run("mkdir -p data", shell=True)

subprocess.run("rm data/*", shell=True)

subprocess.run([
    "curl", "-o", "data/jailbreaks.json", 
    "https://gist.githubusercontent.com/deadbits/e93a90aa36c9aa7b5ce1179597a6fe3d/raw/30527efde9e1d2b11ede9d9bf205d98c2ba9d550/jailbreaks.json"
    ])

subprocess.run([
    "curl", "-L", "-o", 
    "data/prompt-injection-in-the-wild.zip", 
    "https://www.kaggle.com/api/v1/datasets/download/arielzilber/prompt-injection-in-the-wild"
    ])

subprocess.run([
    "unzip", "data/prompt-injection-in-the-wild.zip",
    "-d", "data"
    ])

subprocess.run("rm data/*.zip",shell=True)

subprocess.run([
    "curl", "-L", "-o",
    "data/jailbreak_prompts.parquet",
    "https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/resolve/main/hackaprompt.parquet?download=true"
])

print("Download completed successfully.")