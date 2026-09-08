import subprocess
import re

commands = [
    "sh reset.sh",
    "python evaluation_voc12.py",
    "mv sio_maps/eval.txt eval_on_{idx}.txt"
]

def update_config(values):
    with open("config_template.py") as f:
        content = f.read()

    updated_config = re.sub(r'ts\s*=\s*\d+', f'ts = {values}', content)

    with open("config.py", "w") as f:
        f.write(updated_config)

for token_idx in range(0, 10):
    update_config(token_idx)
    for cmd in commands:
        subprocess.run(cmd.format(idx=token_idx), shell=True, check=False)
