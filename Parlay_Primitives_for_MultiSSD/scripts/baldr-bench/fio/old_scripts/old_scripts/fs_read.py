from argparse import ArgumentParser
import random
import signal
import subprocess

parser = ArgumentParser()
parser.add_argument("-j", "--jobs", type=int, default=1, 
                    help="Number of threads to use. Should not affect the result as long as block size is large")
parser.add_argument("-b", "--block_size", type=str, default="4M", 
                    help="Block size. fio seems to be constrained to 512kb blocks. Larger values may not make a difference")
parser.add_argument("-n", "--num_ssds", type=int, 
                    help="Number of SSDs to test; use the -s option if you want to specify the SSD you are trying to use")
parser.add_argument("-r", "--round_robin", action="store_true", 
                    help="Works alongside the -n option. Assigns SSDs in a round robin manner so that no single root complex is saturated when the number of SSDs is small")
parser.add_argument("-s", "--ssd", nargs="+", type=str, default=[], 
                    help="Manually specify the SSD number. The script will test /dev/nvme<NUM>n1 for every NUM input. A range such as 0-3 (inclusive) is also acceptable.")
args = parser.parse_args()

num_jobs = args.jobs
block_size = args.block_size

ssd_names = [f"/mnt/ssd{i}/fio" for i in range(0, 30)]

root_complex_list = [
    ssd_names[0:7], ssd_names[7:14], ssd_names[14:22], ssd_names[22:30]
]
# for lst in root_complex_list:
#     random.shuffle(lst)

if args.num_ssds is not None:
    ssd_count = args.num_ssds

    def round_robin(lst: list[list[int]], count: int) -> list:
        num_ssds = sum(len(l) for l in lst)
        assert num_ssds >= count, f"{num_ssds} SSDs found; {count} SSDs needed"
        result = []
        while count > 0:
            for ssd_list in lst:
                if len(ssd_list) > 0:
                    result.append(ssd_list.pop())
                    count -= 1
                if count == 0:
                    break
        return result

    if args.round_robin:
        print("Using round robin to select SSDs")
        ssd_names = round_robin(root_complex_list, ssd_count)
    else:
        ssd_names = ssd_names[0:ssd_count]
elif len(args.ssd) > 0:
    ssd_numbers = []
    for ssd_str in args.ssd:
        if "-" in ssd_str:
            start, end = list(map(int, ssd_str.split("-")))
            ssd_numbers.extend([i for i in range(start, end + 1)])
        else:
            ssd_numbers.append(int(ssd_str))
    ssd_names = [ssd_names[num] for num in ssd_numbers]
else:
    print("Need to either specify # of SSDs or the nvme device number of each SSD.")
    exit(0)

file_string = ":".join(name.replace(":", r"\:") for name in ssd_names)
print("Benchmarking with files " + file_string)

keep_running = True
def interrupt(sig, frame):
    global keep_running
    keep_running = False
    print("fio will be interrupted.")

signal.signal(signal.SIGINT, interrupt)

p = subprocess.Popen(
    ["fio", "--name=fs", "--filename=" + file_string, "--filesize=10g",
    "--rw=read", "--bs=" + block_size, "--direct=1", "--overwrite=0", 
    "--numjobs=" + str(num_jobs), "--iodepth=64", 
    "--time_based=1", "--runtime=20", 
    "--ioengine=io_uring", "--registerfiles",
    "--gtod_reduce=1", "--group_reporting"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
output, _ = p.communicate()

while True:
    try:
        p.wait(timeout=0.1)
        break
    except subprocess.TimeoutExpired():
        if not keep_running:
            p.send_signal(signal.SIGINT)
            p.wait()
            break

print(output.decode("utf-8"))