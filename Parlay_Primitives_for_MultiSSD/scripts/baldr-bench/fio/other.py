from experiment import run_experiment

def max_4k_reads():
    for jobs in [32, 40, 50, 64, 80, 96]:
        args = ['-j', jobs, '-b', '4k', '-n', 28, '-r']
        print(f"Jobs: {jobs}")
        # blog author mentioned lots of extra options to squeeze more performance out of it
        # we did not find it necessary because performance remains unchanged
        print(run_experiment([str(arg) for arg in args], []))


def no_direct_io():
    for jobs in [4, 8, 16, 32, 64]:
        print(f"Jobs: {jobs}")
        print(run_experiment(['-j', jobs, '-b', '4m', '-n', 28, '-r', '--no-direct']))


if __name__ == "__main__":
    no_direct_io()
