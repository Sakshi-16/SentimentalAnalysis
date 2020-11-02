import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ijson
import xlrd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


pd.set_option("display.width",1000)
pd.set_option("display.max_columns",20)

#------got business that must be into restaurants business--------
# businessDF = pd.read_excel(r'F:/ML_Projects/Restaurant Reviews/yelp_business.xlsx')
# businessDF = businessDF.dropna()
# businessDF = businessDF[businessDF['categories'].str.contains('Restaurant',case=False,na=False)]
# businessDF.to_excel('F:/ML_Projects/Restaurant Reviews/yelp_Restaurants.xlsx')

# df = pd.read_csv(path)
# df.drop['date','type','user_id','review_id']
# restro = pd.read_excel(r'F:/ML_Projects/Restaurant Reviews/yelp_Restaurants.xlsx')

review_path = "F:/ML_Projects/Restaurant Reviews/yelp_reviews.xlsx"
df = pd.read_excel(review_path)
df.drop(['business_id'],axis=1,inplace=True)
# df = df[df.business_id.isin(restro['business_id'])]
# print(df.head())

#-----------let's see the count for each stars/rating-----------
# plt.hist(df['stars'])
# plt.xticks(df['stars'].unique())
# plt.show()

#--------------------make all lowercase-----------
df['text'] = df['text'].str.lower()

#---------------------remove punctuations---------
punctuations = ("?", ".", ";", ":", "!",'"',',')
def remove_punct(text):
    text = ''.join(w for w in text if w not in punctuations)
    return text
df['text'] = df['text'].apply(remove_punct)


#-------------------remove helping verbs/articles/stop words-----------------------
stopwords = stopwords.words('english')
stopwords += ['really','place','one','food','service','order','ordered']
def removeStopwords(text):
    text = text.split(' ')
    text = ' '.join(w for w in text if w not in stopwords)
    return text

df['text'] = df['text'].apply(removeStopwords)
#
# positive = df[df.stars.isin([4,5])]
# negative = df[df.stars.isin([1,2])]
#
# ttext = ' '.join(t for t in positive['text'])
# wordCloud = WordCloud().generate(ttext)
#
# plt.imshow(wordCloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig('wordcloud12.png')
# plt.show()
#
# ttext = ' '.join(t for t in negative['text'])
# wordCloud = WordCloud().generate(ttext)
#
# plt.imshow(wordCloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig('wordcloud13.png')
# plt.show()

#we can see from here, that in positive reviews frequent words are good, great, love (positive adjectives)
#but in negative reviews, there's no bad words juts less frequency of good words




# print(positive.shape)
# print(negative.shape)
#80% reviews are positive, high chances that our model will be biased. There's not much data of negative reviews to train our model efficiently

#--------convert 4,5 stars review as positive and 3,2,1 as negative
stars_map = {1:-1, 2:-1, 3:-1, 4:1, 5:1}
df['stars'] = df['stars'].replace(stars_map)

# -----------------make a bag of words------------
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
bagOfWords = vectorizer.fit_transform(df['text'])
print(bagOfWords.shape)


#------split the dataset-------
x = bagOfWords
y = df['stars']
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.33,random_state=42)


#--------------fit it into model--------------
lr = LogisticRegression(max_iter=2000)
lr.fit(xTrain,yTrain)
yPredict = lr.predict(xTest)
print(confusion_matrix(yPredict,yTest))
print(classification_report(yPredict,yTest))