import os
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def proccess_text(text):
    text = ''.join([character for character in text if not character.isdigit()])  # remove numbers

    text = ''.join(character for character in text if
                   character.isalnum() or character == ' ')  # remove special charactars ==' ' 3shan kan byms7 el space ely ben el kalemat
    text = " ".join(text.split())  # remove extra spaces

    return text


posetive_path = "D:\\Universty\\Last year\\2d Term\\NLP\\txt_sentoken\\pos"
negative_path = "D:\\Universty\\Last year\\2d Term\\NLP\\txt_sentoken\\neg"


reviews = []

for root, dirs, files in os.walk(posetive_path):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()

                text = proccess_text(text)
                reviews.append(text)

for root, dirs, files in os.walk(negative_path):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                text = proccess_text(text)

                reviews.append(text)


tfidf_vectorizer = TfidfVectorizer()

tfidf = tfidf_vectorizer.fit_transform(reviews)

target = []
for i in range(0, 2000):
    if i < 1000:
        target.append(1)
    else:
        target.append(0)

X_train, X_test, y_train, y_test = train_test_split(tfidf, target, test_size=0.2, random_state=33)

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

model = LogisticRegression(random_state=0).fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

#print("Training-accuracy:   %0.3f" % train_acc)

#print("Testing-accuracy:   %0.3f" % test_acc)


naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)


y_pred = naive_bayes_classifier.predict(X_test)
y_pred_train = naive_bayes_classifier.predict(X_train)



# compute the performance measures
score1 = metrics.accuracy_score(y_test, y_pred)
#print("Testing-accuracy:   %0.3f" % score1)


score2 = metrics.accuracy_score(y_train, y_pred_train)
#print("Training-accuracy:   %0.3f" % score2)

input = input("Please enter your review")
input = proccess_text(input)
tfidf_input = tfidf_vectorizer.transform([input])


output =naive_bayes_classifier.predict(tfidf_input)

if output[0] == 0:
    print("Negative")
else :
    print ("Positive")


