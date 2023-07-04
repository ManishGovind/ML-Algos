
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def main():
     df=pd.read_csv('diabetes_temp.csv')
     pd.set_option('display.max_columns', None) # to display all columns
     
     # remove null data rows and dulicate rows
     dfn = df.dropna()
   
     dfc = dfn.drop_duplicates()
     
     dfc.Age.plot(color="blue",kind="hist", bins=20)
     plt.xlabel("Age")
     plt.ylabel("frequency")
     plt.show()

     dfc.hist(column='Age', by='Outcome') # histogram by group
     plt.show()

     df2= dfc[["Age","Glucose"]]
     df2.plot(kind='hist',
     alpha=0.7,
     bins=30,
     title='Histogram Of Age,Glucose',
     rot=45,
     grid=True,
     figsize=(10,8),
     fontsize=15,
     color=['#ff0000', '#00ff00'])
     plt.xlabel('Age/Glucose')
     plt.ylabel("Count");
     plt.show()

     sns.countplot(x="Outcome", data=dfc)
     plt.show()
     print(dfc.Outcome.value_counts())

     counts=dfc.Pregnancies.value_counts().values[0:9]
     labels=dfc.Pregnancies.value_counts().index[0:9]
     colors=["green","pink","yellow","purple","grey","blue","plum","orange",
     "red"]
     plt.pie(counts,data=dfc,labels=labels,colors=colors,radius=1.5)
     plt.show()

     df2=(dfc.columns[0:]) 
     dfc.hist(df2,bins=30, figsize=(10,15))
     plt.show()
     sns.heatmap(dfc[df2].corr(),annot=True)
     plt.show()




if __name__ == "__main__" :
    sys.exit(int(main()or 0))


