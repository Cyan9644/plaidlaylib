from random import randrange
import subprocess
import signal
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction

def run_experiment(cmd_args: list[str], extra_fio_args: list[str] = [], silent=False) -> str:

    cmd_args = [str(s) for s in cmd_args]

    # taken from the result of `ls -l /dev/disk/by-path`
    ssd_names_raw_block_device = [
        "/dev/disk/by-path/pci-0000:01:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:02:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:03:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:04:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:05:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:06:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:07:00.0-nvme-1",
        # this is the operating system "/dev/disk/by-path/pci-0000:08:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:23:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2a:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2b:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2c:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2d:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2e:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:2f:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:41:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:42:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:43:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:44:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:45:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:46:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:47:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:48:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:61:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:62:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:63:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:64:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:65:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:66:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:67:00.0-nvme-1",
        "/dev/disk/by-path/pci-0000:68:00.0-nvme-1",
    ]

    ssd_names_with_file_system = [f"/mnt/ssd{i}/fio" for i in range(0, 30)]

    # list of SSDs on each root complex
    def get_root_complex_list(ssd_list):
        return [
            ssd_list[0:7], ssd_list[7:14], ssd_list[14:22], ssd_list[22:30]
        ]

    parser = ArgumentParser()
    parser.add_argument("--rw", type=str, default="read",
                        help="fio workload (read/write/randread/randwrite/rw)")
    parser.add_argument("-j", "--jobs", type=int, default=1, 
                        help="Number of threads to use. Should not affect the result as long as block size is large")
    parser.add_argument("-b", "--block_size", type=str, default="4M", 
                        help="Block size. fio seems to be constrained to 512kb blocks. Larger values may not make a difference")
    parser.add_argument("-n", "--num_ssds", type=int, 
                        help="Number of SSDs to test; use the -s option if you want to specify the SSD you are trying to use")
    parser.add_argument("-r", "--round_robin", action="store_true", 
                        help="Works alongside the -n option. Assigns SSDs in a round robin manner so that no single root complex is saturated when the number of SSDs is small")
    parser.add_argument("-s", "--ssd", nargs="+", type=str, default=[], 
                        help="Manually specify the SSD number. See the list of SSDs for which ")
    parser.add_argument("--file_system", "--fs", action="store_true",
                        help="Use mounted SSDs via the file system instead of raw block devices.")
    parser.add_argument("--latency", action="store_true",
                        help="Measure the latency of io requests. This option will slightly increase CPU overhead.")
    parser.add_argument("--file_size", type=str, default="100g",
                        help="Size of the file to be used in fio's tests.")
    parser.add_argument("--direct", action=BooleanOptionalAction, default=True,
                        help="Whether to use O_DIRECT in tests.")
    parser.add_argument("--depth", type=int, default=64,
                        help="io_uring queue depth.")
    parser.add_argument("--runtime", type=int, default=20,
                        help="Runtime of the script (in seconds).")
    cmd_args = parser.parse_args(cmd_args)

    workload = cmd_args.rw
    num_jobs = cmd_args.jobs
    block_size = cmd_args.block_size

    use_file_system = cmd_args.file_system
    if use_file_system:
        ssd_names = ssd_names_with_file_system
    else:
        ssd_names = ssd_names_raw_block_device
        import os
        if os.getuid() != 0:
            raise RuntimeError("Need root privilages to read from raw block devices")

    if cmd_args.num_ssds is not None:
        ssd_count = cmd_args.num_ssds

        def round_robin(lst: list[list[int]], count: int) -> list:
            num_ssds = sum(len(l) for l in lst)
            assert num_ssds >= count, f"{num_ssds} SSDs found; {count} SSDs needed"
            result = []
            while count > 0:
                for ssd_list in lst:
                    if len(ssd_list) > 0:
                        result.append(ssd_list.pop(randrange(0, len(ssd_list))))
                        count -= 1
                    if count == 0:
                        break
            return result

        if cmd_args.round_robin:
            ssd_names = round_robin(get_root_complex_list(ssd_names), ssd_count)
        else:
            ssd_names = ssd_names[0:ssd_count]
    elif len(cmd_args.ssd) > 0:
        ssd_numbers = []
        for ssd_str in cmd_args.ssd:
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

    keep_running = True
    def interrupt(sig, frame):
        global keep_running
        keep_running = False
        print("fio will be interrupted.")

    signal.signal(signal.SIGINT, interrupt)

    fio_args = ["fio"]
    fio_args.extend([
        "--name=all_ssd", "--filename=" + file_string, f"--filesize={cmd_args.file_size}",
        f"--rw={workload}", "--bs=" + block_size, "--group_reporting", "--time_based=1", f"--runtime={cmd_args.runtime}",
        f"--numjobs={num_jobs}", "--overwrite=0"
    ])

    if not use_file_system and "write" not in workload:
        fio_args.append("--readonly")

    if not cmd_args.latency:
        fio_args.append("--gtod_reduce=1")

    fio_args.append(f"--direct={1 if cmd_args.direct else 0}")

    fio_args.extend(["--ioengine=io_uring", "--registerfiles"])
    if not use_file_system:
        # this option requires root privilages and does not measurably improve performance unless the CPU is bottlenecked
        fio_args.append("--fixedbufs")
    fio_args.append(f"--iodepth={cmd_args.depth}")
    fio_args.extend(extra_fio_args)

    if not silent:
        print("Fio command: " + " ".join(fio_args))

    p = subprocess.Popen(
        fio_args,
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

    return output.decode("utf-8")


if __name__ == "__main__":
    print(run_experiment(argv[1:]))
