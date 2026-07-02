from experiment import run_experiment
from converter import convert_group

job_numbers = [2, 4, 8, 16]
block_sizes = ['16k', '64k', '256k', '1m', '4m', '16m']

results = []
summaries = []
for num_jobs in job_numbers:
    summaries.append(f"j={num_jobs}")
    for size in block_sizes:
        result = run_experiment(['-j', str(num_jobs), '-b', size, '-n', '28', '-r', '--latency'],
                                silent=True)
        results.append(result)
        summaries.append(",".join([size] + list(convert_group(result, measure_cpu=True).values())))
    summaries.append("")
with open("results/ssd_block_size.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summaries))
    f.write("\n\n")
    f.write("\n".join(results))
