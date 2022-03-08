import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report



TheData = pd.read_csv("ListOfNews.csv")

TheData.head(10)
TheData.info()
TheData.shape
TheData["label"].value_counts()
labels = TheData.label
labels.head(10)

x_train,x_test,y_train,y_test = train_test_split(TheData["text"],labels,test_size=0.4,random_state = 7)
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train=vectorizer.fit_transform(x_train)
tfidf_test=vectorizer.transform(x_test)

passive= PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)
y_pred=passive.predict(tfidf_test)

matrix = confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])
matrix
sns.heatmap(matrix,annot=True)
plt.show()

accuracy = accuracy_score(y_test,y_pred)

report = classification_report(y_test,y_pred)

print(report)