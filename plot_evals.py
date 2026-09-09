import os
import re
import matplotlib.pyplot as plt

pattern = r" Best mIoU:\d*\.?\d+"
max_ts = 10

values_range = range(1, 61, 10)
if __name__ == "__main__":
    accuracy_list = []
    for i in values_range:
        eval_file = f"eval_on_{i}.txt"
        with open(eval_file) as f:
            content = f.read()
        result_substring = re.search(pattern, content).group().replace("Best mIoU:", "")
        accuracy_list.append(
            float(result_substring)
        )
    
    plt.scatter(list(values_range), accuracy_list)
    plt.xlabel("timestamp")
    plt.ylabel("accuracy")
    plt.savefig("results.png")
    plt.close()