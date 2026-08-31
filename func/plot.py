import os
import matplotlib.pyplot as plt



def VisAUROC(tpr, fpr, AUROC, method_name, file_name="CoQA"):
    if "coqa" in str(file_name).lower():
        file_name = "CoQA"
    elif "nq" in str(file_name).lower():
        file_name = "NQ"
    elif "trivia" in str(file_name).lower():
        file_name = "TriviaQA"
    elif "squad" in str(file_name).lower():
        file_name = "SQuAD"
    plt.plot(fpr, tpr, label="AUC-{}=".format(method_name)+str(round(AUROC,3)))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('ROC Curve on {} Dataset'.format(file_name), fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    fig_dir = "./Figure"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "AUROC_{}.png".format(file_name)), dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    pass
