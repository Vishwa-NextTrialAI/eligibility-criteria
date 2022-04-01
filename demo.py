import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
df= pd.read_csv('/Users/VishwaPrasad/Desktop/criteria_10k.csv')
print(df.shape)
print(df.isnull().sum())
print(df[df.criteria.isnull()])
df.drop([842, 1081, 2783, 3990, 6305, 6649], axis=0, inplace=True)
df.drop_duplicates(inplace=True)
print(df.shape)


print(df['criteria'].apply (lambda x: len(x.split(' '))).sum())

special_char_remover=re.compile('[/(){}\[\]\|@,;]')
extra_sym_remover=re.compile('[^0-9a-z #+_]')
stopwords=set(stopwords.words('english'))

def clean_text(text):
    text=text.lower()
    text=special_char_remover.sub(' ',text)
    text=extra_sym_remover.sub('',text)
    text=' '.join(word for word in text.split() if word not in stopwords)
    return text

df['criteria']=df['criteria'].apply(clean_text)

criterias= df['criteria'].values.astype('U')
vectorizer=TfidfVectorizer(stop_words='english')
features=vectorizer.fit_transform(criterias)

k=6
model=KMeans(n_clusters=k,init='k-means++',max_iter=1000,n_init=1)
model.fit(features)
# print(features)
df['cluster']=model.labels_
print(df.tail())
df['slno'] = np.arange(df.shape[0])
df.set_index('slno')
print(df.head())

# clusters=df.groupby('cluster')
# for cluster in clusters.groups:
#     f=open('cluster'+str(cluster)+'.csv','w')
#     data=clusters.get_group(cluster)['criteria']
#     f.write(data.to_csv(index_label='slno'))
#     f.close()
# print(df.head())
cluster0=df[model.labels_==0]
cluster0.to_csv('cluster0.csv',index=False)
cluster1=df[model.labels_==1]
cluster1.to_csv('cluster1.csv',index=False)
cluster2=df[model.labels_==2]
cluster2.to_csv('cluster2.csv',index=False)
cluster3=df[model.labels_==3]
cluster3.to_csv('cluster3.csv',index=False)
cluster4=df[model.labels_==4]
cluster4.to_csv('cluster4.csv',index=False)
cluster5=df[model.labels_==5]
cluster5.to_csv('cluster5.csv',index=False)


print("cluster centroids:")
order_centroids=model.cluster_centers_.argsort()[:,::-1]
terms=vectorizer.get_feature_names()
for i in range (k):
    print("cluster %d:" %i)
    for j in order_centroids[i,:25]:
        print('%s' %terms[j])
    print('------------------')

# import os
#
# os.rename('cluster0.csv', 'demographic.csv')