# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Here's a simple four-sentence algorithm for implementing K-Means Clustering for customer segmentation:
1. Initialize by selecting 𝑘
2. k cluster centroids randomly from the dataset.
3. Assign each customer to the nearest centroid based on a distance measure (e.g., Euclidean distance).
4. Update each centroid to the mean position of all customers assigned to it.
5. Repeat steps 2 and 3 until centroids no longer change significantly, then use the resulting clusters to segment the customers.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Sukhmeet Kaur G
RegisterNumber: 2305001032
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))

def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:
![Screenshot 2024-11-06 194657](https://github.com/user-attachments/assets/22694b62-7d55-4aba-ad7c-16aacbf37fce)
![Screenshot 2024-11-06 194708](https://github.com/user-attachments/assets/d103dd4d-ac61-41a3-8888-1d4b66038cfb)
![Screenshot 2024-11-06 194716](https://github.com/user-attachments/assets/87bde229-e22e-45b2-b814-2d5d9b896b0b)
![Screenshot 2024-11-06 194726](https://github.com/user-attachments/assets/9a31d475-1f2b-423f-a483-29ec40c22f18)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
