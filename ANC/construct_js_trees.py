import subprocess
import glob
from natsort import natsorted

path_names = natsorted(glob.glob("validation_js_programs/*"))

result = "["

for i, path_name in enumerate(path_names):
    if i % 100 == 0:
        print(i)

    proc = subprocess.Popen(['acorn', path_name], stdout=subprocess.PIPE)
    output = proc.stdout.read().decode('ascii')

    result += output

    if path_names[-1] != path_name:
        result += ", "

result += "]"

with open("validation_JS.json", "w") as f:
    f.write(result)