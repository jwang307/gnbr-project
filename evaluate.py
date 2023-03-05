import sklearn.metrics as metrics
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')


def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average='micro')


def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)


if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except IndexError:
        print("Please provide a path to the csv file.")
        sys.exit(1)

    record = pd.read_csv(path)
    y_true = list(record['labels'])
    y_pred = list(record['preds'])

    print(f1_score(y_true, y_pred))
    print(auc(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    sns.set(font_scale=2.5)
    label_font = {'size': '18'}  # Adjust to fit

    cm = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True))
    fig = cm.get_figure()
    cm.set_xlabel('Predicted labels', fontdict=label_font)
    cm.set_ylabel('Observed labels', fontdict=label_font)
    cm.tick_params(labelsize=12)
    fig.tight_layout()

    fig.savefig("predictions/confusion_matrix.png")