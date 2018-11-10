import subprocess
import glob
from natsort import natsorted

processed_files = set()
num_processed_files = 0
list_of_jsons = []

while num_processed_files != 100000:
    path_names = natsorted(glob.glob("*.js"))

    for path_name in path_names:
        if path_name in processed_files:
            continue

        path_number = int(path_name[:-3])

        proc = subprocess.Popen(['acorn', path_name], stdout=subprocess.PIPE)
        output = proc.stdout.read().decode('ascii')

        list_of_jsons.append((path_number, output))

        if num_processed_files % 100 == 0:
            print(num_processed_files)

        num_processed_files += 1
        processed_files.add(path_name)

result = "[" + ", ".join(map(lambda pair: pair[1], sorted(list_of_jsons, key=lambda pair: pair[0]))) + "]"

with open("../training_JS.json", "w") as f:
    f.write(result)
