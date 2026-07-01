import sys
import random

def main():
    ssd_nums = [i for i in range(0, 30)]
    ssd_list = [ssd_nums[0:7], ssd_nums[7:14], ssd_nums[14:22], ssd_nums[22:30]]
    if len(sys.argv) < 2:
        print("Insufficient arguments")
    n = int(sys.argv[1])
    cur_root_complex = random.randint(0, len(ssd_list) - 1)
    result: list[int] = []
    while n > 0:
        lst = ssd_list[cur_root_complex]
        if len(lst) > 0:
            result.append(lst.pop(random.randint(0, len(lst) - 1)))
            n -= 1
        cur_root_complex += 1
        cur_root_complex %= len(ssd_list)
    print(",".join(str(num) for num in result))


if __name__ == "__main__":
    main()