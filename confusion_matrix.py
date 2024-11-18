import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#
with open(f'testing_dataset.pkl', 'rb') as f:
    _, label = pickle.load(f)


actual_y = np.array(label).reshape(-1, 1)

with open(f'label_predict_by_model_fold_5.pkl', 'rb') as f:
    predicted = pickle.load(f)
print(pd.unique(predicted))

predicted = np.array(predicted).reshape(-1, 1)

print(predicted.shape)

target_names = ['acceptable 0', 'unacceptable 1']
print(classification_report(actual_y, predicted, target_names=target_names))

# accuracy
results = metrics.accuracy_score(actual_y, predicted)
print("Test Accuracy: ", results)

confusion_matrix = confusion_matrix(actual_y, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot(cmap=plt.cm.Blues)
for text in cm_display.text_.ravel():
    text.set_fontsize(18)
plt.ylabel('True label',fontsize=14)
plt.xlabel('Predicted label',fontsize=14)
plt.show()
