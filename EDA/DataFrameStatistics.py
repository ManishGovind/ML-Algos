
import pandas as pd 
import numpy
import sys 

def main() :
    df = pd.read_csv('diabetes_temp.csv')
    pd.set_option('display.max_columns', None) 
    print(df.shape)

    first_nrows = df.head()
    last_nrows = df.tail()

    #print(df.info()) 
    df.dtypes # data types of each column
    print("Null values\n",df.isna().sum())
    #print("Duplicate values", df.duplicated().sum())  Return the sum of the values over the requested axis.
    dfn = df.dropna()

    df_clean = dfn.drop_duplicates()
    #print(df_clean.shape)

    column_headers = list(df.columns.values)
    print("Column Headings :", column_headers)

    data_columns = column_headers[0:-1]
    #print(data_columns)

    data_corr = df_clean[data_columns].corr() # correlation between columns
    #print(data_corr)
     # determine the highest correlations for a feature
    corr_data = df_clean.corr().abs()
    cbmi = corr_data["BMI"]
    #print(cbmi)
    corr_bmi = cbmi.sort_values(ascending = False)
    #print("sorted correlations--------")
    #print(corr_bmi)
    description_of_data_frame =  df_clean.describe() # details of basic stats
    no_of_unique_records =  df_clean.nunique(axis=0, dropna=True) # count nof unique rows or columns
    
    print(no_of_unique_records)
    #print(df_clean['Pregnancies'].unique()) # print unique values of Pregnancies
    # combining columns from one dataframe to another new dataframe
    seriesBMI = df["BMI"]
    seriesGlucose = df["Glucose"]
    seriesOutcome = df["Outcome"]
    dfnew = pd.concat([seriesBMI,seriesGlucose, seriesOutcome], axis=1)
    #print(dfnew.head())

    df_filtered = dfnew[dfnew['Glucose']<= 180]
    #print(len(df_filtered))
    #print(df_filtered.head(728))
    # change Outcome colum 1 and 0 classes to 1 and 2
    dfnew['Outcome'] = dfnew['Outcome'].map(lambda x: 1 if x ==1 else 2)
    

    # access a particular row, column data in a dataframe
    d1 = dfnew.iloc[25,1] 
    
    # select 10-13 rows into a new dataframe
    d2 = dfnew.iloc[10:14,:]
  
    # select 10-13 rows into a new dataframe and reset index
    d3 = dfnew.iloc[10:14,:].reset_index()
    print(d3)
    
    # select 10-13 rows into a new dataframe and reset index
    # drop=True to avoid old index being kept
    d3 = dfnew.iloc[10:14,:].reset_index(drop=True)
    print(d3)

if __name__ == "__main__":
    sys.exit(int(main() or 0))





