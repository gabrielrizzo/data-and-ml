from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

composto1 = [1, 1, 1]
composto2 = [0, 0, 0]
composto3 = [1, 0, 1]
composto4 = [0, 1, 0]
composto5 = [1, 1, 0]
composto6 = [0, 0, 1]

training_data = [composto1, composto2, composto3, composto4, composto5, composto6]
training_labels = ['S', 'N', 'S', 'N', 'S', 'S']

model = LinearSVC()
model.fit(training_data, training_labels)

control = [[1, 0, 0]]
control_label = ['S']
prediction = model.predict(control)
accuracy = accuracy_score(control_label, prediction)

print(accuracy * 100)