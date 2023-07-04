
import sys
import pandas as pd 
import math
import numpy  as np 


def gaussian_probab(x, mean_value , var_value):
    res = 1 
    for i in range(0,len(x)):
        num =  math.exp(- (( x[i] - mean_value[i])**2 / (2 * (var_value[i]**2))))
        res *= (1/(math.sqrt(2*math.pi)*var_value[i]))*num
        
    return res 

def main ( ):
    #---------------Iris Dataset-------------
    #read the dataset 
    df = pd.read_csv("./iris.csv")
    #---randomize data
    dfrandom = df.sample(frac=1, random_state=1119).reset_index(drop=True)
    #  convert the data  first 4 cols to float
    df1 = dfrandom.iloc[:,0:4].astype(float)
    print(df1)
    #---separate out the last column
    df2 = dfrandom.iloc[:,4]
    #---combine the 4 numerical columns and the ast column that has the flower category
    dfrandom = pd.concat([df1,df2],axis=1)
   
    #split data 
    df_train = dfrandom.iloc[0:100, :]
    
    df_test = dfrandom.iloc[100:, :]
   


    df_setosa = df_train[df_train['species'] == 'setosa']
    df_versicolor = df_train[df_train['species'] == 'versicolor']
    df_virginica = df_train[df_train['species'] == 'virginica']

    mean_setosa = df_setosa.iloc[:,0:4].mean(axis = 0 ) 
    mean_versicolor = df_versicolor.iloc[:,0:4].mean(axis = 0 ) 
    mean_virginica = df_virginica.iloc[:,0:4].mean(axis = 0 ) 
    
    var_setosa = df_setosa.iloc[:,0:4].var(axis=0)
   
    var_versicolor = df_versicolor.iloc[:,0:4].var(axis=0)
    var_virginica = df_virginica.iloc[:,0:4].var(axis=0)
    

    acc_count = 0
    
    for i in range(0,len(df_test)):
        x = df_test.iloc[i,0:4].values
        prob_setosa = gaussian_probab(x,mean_setosa.values,var_setosa.values)
        prob_versicolor = gaussian_probab(x,mean_versicolor.values,var_versicolor.values)
        prob_virginica = gaussian_probab(x,mean_virginica.values,var_virginica.values)
        probs = np.array([prob_setosa,prob_versicolor,prob_virginica])

        max_prob = probs.argmax(axis=0)

        print("probs " , max_prob)
        if (df_test.iloc[i,4] == 'setosa' ):
            gt = 0
        elif  (df_test.iloc[i,4] == 'versicolor' ):
            gt = 1
        elif (df_test.iloc[i,4] == 'virginica' ):
            gt = 2
        
        if max_prob == gt :
            acc_count = acc_count + 1

        print(prob_setosa,' ', prob_versicolor,' ', prob_virginica,' class=',df_test.iloc[i,4])
    print('classification accuracy =', acc_count/len(df_test)*100) 


   



if __name__ == '__main__':
    sys.exit(int(main()or 0))