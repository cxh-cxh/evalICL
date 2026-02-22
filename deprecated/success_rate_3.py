import json
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def map_to_category(difficulty):
    if 0 <= difficulty <= 2:
        return 2
    elif 8 <= difficulty <= 10:
        return 0
    elif 3 <= difficulty <= 7:
        return 1
    else:
        return None
    
def trans(difficulty):
    if difficulty == "easy":
        return 0
    elif difficulty == "hard":
        return 2
    elif difficulty == "medium":
        return 1
    else:
        return None


def load_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    results = [map_to_category(int(q.get('success_rate', 0.0).split('/')[0])) for q in queries]
    first_success = [int(q.get('first_success', 0)) for q in queries]
    return results, first_success

def load_difficulties(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        difficulties = json.load(file)
    return [trans(d) for d in difficulties]



_queries, _first_success = load_queries("./results/20250918_150537_gpt-4o_3/queries.jsonl")
_difficulties = load_difficulties("./results/20250918_150537_gpt-4o_3/difficulties.json")


c = list(zip(_queries, _difficulties, _first_success))
np.random.shuffle(c)
# c = c[:30]
queries, difficulties, first_success = zip(*c)
success_rate = 0


valid_idx = [i for i, d in enumerate(difficulties) if d != None]
queries_filtered = [queries[i] for i in valid_idx]
difficulties_filtered = [difficulties[i] for i in valid_idx]

queries_filtered = np.array(queries_filtered, dtype=int)
difficulties_filtered = np.array(difficulties_filtered, dtype=int)

for i in range(len(queries_filtered)):
    if queries_filtered[i] == difficulties_filtered[i]:
        success_rate += 1
success_rate /= len(queries_filtered)
print(f"Success Rate: {success_rate:.2%}")


cm = confusion_matrix(queries_filtered, difficulties_filtered,labels=[0,1,2])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['easy', 'medium', 'hard'], yticklabels=['easy', 'medium', 'hard'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Success Rate: {success_rate:.2%}')
plt.show()
