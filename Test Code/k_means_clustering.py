import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Data\\features_30_sec.csv')
df = df.drop(['label'], axis=1)
df = df.set_index(df['filename'])
df = df.drop(['filename'], axis = 1)
df = df.drop(['length','rms_var',
              'zero_crossing_rate_var','harmony_mean',
              'perceptr_mean'], axis=1)

# plt.gcf().set_size_inches(15,15)
# cmap=sns.diverging_palette(500,10,as_cmap=True)
# sns.heatmap(df.corr(),center=0,annot=False,square=True)

# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
# print(scaled_df)

kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(scaled_df)

pd.options.display.max_columns = 13

predictions = k_fit.labels_
type(predictions)
scaled_df['clusters'] = predictions

pca = PCA(n_components=2)
pca_data = pd.DataFrame(pca.fit_transform(scaled_df.drop(['clusters'],axis=1)),columns=['PC1','PC2'], index=scaled_df.index)
pca_data['clusters']=predictions

plt.figure(figsize=(10,10))
sns.scatterplot(data=pca_data,x='PC1',y='PC2',hue='clusters',palette='Set2' , alpha = 0.9)
plt.title('Music Recommendation after PCA')
# dist = sqrt((x1-x2)^2 + (y1 - y2)^2)
plt.show()
print(pca_data)