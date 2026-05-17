from plot_factory import *

func_names = ["linear", "mult_sqrt", "l2"]

for func_name in func_names:
    with open(f"./collate/result_{func_name}.json") as f:
        data_ = json.load(f)

    data = data_["pi0_t10003_sim_full"]
    data["length"] = len(data["new"])
    data["old_scores"] = np.array([data["old"]])
    data["scores"] = np.array([data["new"]])
    data["easy"] = [data["easy"]]
    data["medium"] = [data["medium"]]
    data["hard"] = [data["hard"]]
    axs = get_subplots(3)
    difficulty_plot(axs[0], data)
    score_plot(axs[1], data)
    variance_plot(axs[2], data)
    savefig(f"pi0_t10003_sim_full_{func_name}.png")
