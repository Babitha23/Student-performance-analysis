#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Step 1 - Import all the required libraries for data analysis
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


pd.options.mode.chained_assignment = None #to suppress chained assignment warning

#Step2 - Load the data from .csv files into pandas dataframes
df1 = pd.read_csv('student-mat.csv',index_col=False,sep=';')
df2 = pd.read_csv('student-por.csv',index_col=False,sep=';')


#Step3 - Checking for shapes of datasets and Looking out for missing values
def explore():
    print("Maths dataset has: '%d' observations & '%d' Attributes including target" % df1.shape + f" and has '{df1.isnull().values.sum()}' missing values ")
    print("Portuguese dataset has: '%d' observations & '%d' Attributes including target" % df2.shape + f" and has '{df2.isnull().values.sum()}' missing values ")


#Step4 - Merging the two datasets related to maths and portuguese grades.
df3 = pd.concat([df1,df2],axis=0) #Concatinating the datasets vertically


#Step5 - Removing duplicate data (There are 382 students that belong to both datasets)
df = df3.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])


df_dummy = df.copy() #creating a copy of data before converting categorical to numeric


#Step6 - Convert categorical features to numeric using OrdinalEncoder
enc = OrdinalEncoder()
df.loc[:,['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic']] = enc.fit_transform(df[['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic']])


#Based on Final Grades G3, students are being classified into 'Low', 'Medium' and 'High'.
cat = [0, 10.0, 15.0, 20.0]
cat_name = ['low','medium','high']
df.loc[:,'G3_binned']= pd.cut(df['G3'], bins=cat, labels= cat_name,include_lowest=True)


#Step7 - Statistical Analysis for numerical features
def stats_num():
    cate = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities','nursery', 'higher', 'internet', 'romantic', 'G3_binned']
    d1 = pd.DataFrame({"Mean":round(np.mean(df),2)})
    d2 = pd.DataFrame({"Median":df.median()})
    d3 = pd.DataFrame({"Minimum":np.min(df)})
    d4 = pd.DataFrame({"Maximum":np.max(df)})
    d5 = pd.DataFrame({"Variance":round(np.var(df),2)})
    d6 = pd.DataFrame({"Std_Dev":round(np.std(df),2)})
    d7 = pd.DataFrame({"Skewness":round(df.skew(),2)})
    d8 = pd.DataFrame({"Kurtosis":round(df.kurt(),2)})
    dfs=[d1,d2,d3,d4,d5,d6,d7,d8]
    result = pd.concat(dfs, join='outer', axis=1)
    num_stat = result.drop(cate, axis=0)
    return(num_stat)


#Step8 - Statistical Analysis for Categorical features
def stats_cat():
    count = []
    Categories = []
    top = []
    top_count = []
    total_count = []
    labels = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities','nursery', 'higher', 'internet', 'romantic']
    i=0
    for k in labels:
        count.insert(i,df_dummy[k].nunique())
        Categories.insert(i,list(df_dummy[k].unique()))
        top.insert(i,df_dummy[k].value_counts().idxmax())
        top_count.insert(i,df_dummy[k].value_counts().max())
        total_count.insert(i,df_dummy[k].count())
        i+=1
    s1 = pd.Series(count,labels,name="Unique_Count")
    s2 = pd.Series(Categories,labels,name="Categories")
    s3 = pd.Series(top,labels,name="Top_Category")
    s4 = pd.Series(top_count,labels,name="Top_Count")
    s5 = pd.Series(total_count,labels,name="Total_Count")
    df_cat = pd.concat([s1,s2,s3,s4,s5],axis=1)
    return(df_cat)

def initialplots():
    #Checking for correlation between features and class.
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df_dummy.corr(),vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
    plt.show()
    
    #Plotting and comparing features with class
    sns.violinplot(x=df['G3'])
    plt.title("The spread of G3 grades")
    
def barplots():
    
    df_zero = df_dummy[df_dummy['G3']==0]
    sns.catplot(y="school", hue="schoolsup", kind="count", palette="pastel", edgecolor=".6", data=df_zero)
    plt.title("Students with '0' score and their Schools")

    sns.catplot(x="famrel", y="Dalc", kind="bar", data=df_dummy)
    plt.title("Family relationship vs Weekday alcohol consumption")

    sns.catplot(x="famrel", y="Walc", kind="bar", data=df_dummy)
    plt.title("Family relationship vs Weekend alcohol consumption")

    sns.catplot(x="age", y="Walc", kind="bar", data=df_dummy)
    plt.title("Weekend Alcohol consumption with age comparison")

    sns.catplot(x="age", y="Dalc", kind="bar", data=df_dummy)
    plt.title("Weekday Alcohol consumption with age comparison")

    sns.catplot(x="age", y="failures", hue="school", kind="bar", data=df_dummy)
    plt.title("Student age vs their past failures")

    sns.catplot(x="failures", y="G3", kind="bar", data=df_dummy)
    plt.title("Past failures vs G3")
    
def boxplots():
    sns.catplot(x='Fjob', y='G3', data=df_dummy, kind="box", palette='viridis', order=['teacher','health','services','other','at_home'])
    plt.title("Father's job vs Students' G3 score")

    sns.catplot(x='Mjob', y='G3', data=df_dummy, kind="box", palette='cubehelix', order=['teacher','health','services','other','at_home'])
    plt.title("Mother's job vs Students' G3 score")

def pieplot():
    df.groupby(['G3_binned']).sum().plot(kind='pie', y='G3',autopct='%1.0f%%',colors = ['red','coral','green'],
                                    title='Overall performance of students', explode=[0.1,0.0,0.0])

