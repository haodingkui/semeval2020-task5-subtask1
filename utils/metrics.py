from sklearn.metrics import precision_score, recall_score, f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_precision_recall_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "semeval-2020-task5-subtask1":
        return acc_precision_recall_f1(preds, labels)
    else:
        raise KeyError(task_name)
