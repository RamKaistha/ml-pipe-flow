import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('training1.csv')
#print(df)
df.drop(['R','Re','del_p'],axis=1,inplace=True)
from pylab import savefig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

#df1=pd.read_csv('/home/ram/Desktop/R=2.6/training1.csv')
#Y=df1['del_p'].values
#X=df1.drop(['del_p','R','Re'], axis=1)


X_train=np.array(df)




scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train_scaled=pd.DataFrame(data=X_train, columns=['velocity_s','dia1_s','density_s','viscoscity_s','dia2_s'])
X_train_scaled.to_csv('X_train_scaled.csv')
x=np.linspace(1,9326,9326)
x_pd=pd.DataFrame(data=x,columns=['samples'])
x1=np.linspace(9326,2*9326,9326)
x_pd['samples1']=x1

unscaled_vs_scaled=pd.concat([X_train_scaled,df,x_pd],axis=1)

a=unscaled_vs_scaled.plot.scatter(x='samples',y='velocity',color='r',label='Unscaled')

a2=a.twinx()
unscaled_vs_scaled.plot.scatter(x='samples1',y='velocity_s',ax=a2,color='g',label='scaled')
plt.legend()
plt.show()
#plt.scatter(x,unscaled_vs_scaled['viscoscity_s'],label='Scaled')
#plt.scatter(x,unscaled_vs_scaled['viscosity'],label='Unscaled')

#plt.show()




###################################################3PCA_PLOTS##################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_train=PCA()

pca_train.fit(X_train)
X_train_pca=pca_train.transform(X_train)
X_train_scaled_pca=pd.DataFrame(data=X_train_pca, columns=['p1','p2','p3','p4','p5'])
sns.heatmap(X_train_scaled_pca.corr(),annot=True)
plt.show()
scaled_vs_pca=pd.concat([X_train_scaled,X_train_scaled_pca],axis=1)

x=np.linspace(1,9326,9326)
sns.scatterplot(scaled_vs_pca['dia2_s'],scaled_vs_pca['p3'],data=scaled_vs_pca)

#sns.scatterplot(x,scaled_vs_pca['viscoscity_s'],label='scaled')

plt.show()

