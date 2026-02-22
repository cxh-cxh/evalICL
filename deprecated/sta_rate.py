import json
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

def map_to_category(succ):
    if 0 <= succ <= 2 or 9<= succ <=10:
        return 1
    else:
        return 0

def trans(difficulty):
    if difficulty == "unstable":
        return 0
    elif difficulty == "stable":
        return 1



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



_queries, _first_success = load_queries("./results/20250908_131839_gpt-4o_stability_01/queries.jsonl")
_difficulties = load_difficulties("./results/20250908_131839_gpt-4o_stability_01/stabilities.json")


c = list(zip(_queries, _difficulties, _first_success))
np.random.shuffle(c)
# c = c[:30]
queries, difficulties, first_success = zip(*c)
success_rate = 0

for i in range(len(queries)):
    if queries[i] == difficulties[i]:
        success_rate += 1
success_rate /= len(queries)
print(f"Success Rate: {success_rate:.2%}")

fpr, tqr, thresholds =roc_curve(queries, difficulties)
roc_auc = auc(fpr, tqr)

plt.figure()
plt.plot(fpr, tqr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
cm = confusion_matrix(queries, difficulties,labels=[0,1])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['unstable', 'stable'], yticklabels=['unstable', 'stable'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Success Rate: {success_rate:.2%}')
plt.show()
