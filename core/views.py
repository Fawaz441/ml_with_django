import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from django.shortcuts import render

cv = CountVectorizer()
model = joblib.load('model/cybermodel.pkl')
train = open('train','rb')
X_train = pickle.load(train)
cv.fit(X_train)      #countvectorizer needs to be fitted to a vocabulary


def index(request):
    return render(request,'index.html')

def predict(request):
    sentence = [request.POST.get('sentence')]                   #get the sentence 
    sentence_trans = cv.transform(sentence).toarray()           #transform the sentence
    prediction = model.predict(sentence_trans)[0]               #get the prediction
    user_input = sentence[0]
    context = {'result':prediction,'sentence':user_input}
    return render(request,'result.html',context)
