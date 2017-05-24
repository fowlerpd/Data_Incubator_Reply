
# coding: utf-8


import pandas as pd
import numpy as np
import csv
import os
import re

get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})


##Reads text file and generates csv with ID, ASIN, group, salesrank, number of reviews, average review

Input_file = 'amazon-meta.txt'
Output_file = 'amazon_data_full.csv'
header = [
    "Id",
    "ASIN",
    "group",
    "salesrank"]


with open(Output_file, 'w') as output:
    writer = csv.writer(output)
    writer.writerow(['Id', 'ASIN', 'group', 'salesrank', 'number of reviews','average rating'])


    with open(Input_file) as f:
        data = [None] * 6
        for line in f:
            line = line.strip()
            if line == "":
                #print(data)
                writer.writerow([data[0],data[1],data[2],data[3],data[4],data[5]])
                data = [None] * 6
                continue
            parts = line.split(": ",1)
            for i in range(len(header)):
                  if parts[0] == header[i]:
                        data[i] = parts[1]
            if parts[0] == 'reviews':
                a = re.findall(r'[-+]?\d*\.\d+|\d+', parts[1])
                data[4] = a[0]
                data[5] = a[2]
                




file = 'amazon_data_full.csv'
df = pd.read_csv(file);
df = df.dropna(subset = ['number of reviews','average rating'])


df_reviews = df.ix[df['number of reviews'] > 0]
review_counts = df_reviews["average rating"].value_counts(dropna=False).sort_index()


figure = plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
plt.xlabel('Average Rating',fontsize=22)
plt.ylabel('Number of Products',fontsize=22)
plt.title('Distribution of Average Ratings of Amazon Products',fontsize=22,y=1.04)
review_counts.plot(kind='bar',width =1.0)
plt.tight_layout()
plt.savefig("Fig1.pdf",format="pdf")
plt.show()


df_reviews = df.ix[df['number of reviews'] > 10]
review_counts = df_reviews["average rating"].value_counts(dropna=False).sort_index()
avg_review = df_reviews['average rating'].mean()


figure = plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
plt.xlabel('Average Rating (minimum 10 reviews)',fontsize=22)
plt.ylabel('Number of Products',fontsize=22)
plt.title('Distribution of Average Ratings of Amazon Products',fontsize=22,y=1.04)
plt.bar(np.arange(1,5.5,0.5),np.array(review_counts),0.5,color='r',align = 'center')
ys = np.linspace(ymin,ymax,4)
plt.plot(avg_review*np.ones(len(ys)),ys,'--k',linewidth = 2,label = 'Mean Product Rating')
plt.xlim([0.75,5.5])
plt.xticks(np.arange(1,5.5,0.5))
plt.legend(frameon = False,numpoints = 1,fontsize=18,loc =2)
plt.tight_layout()
plt.savefig("Fig2.pdf",format="pdf")


file = 'amazon_data_full.csv'
df = pd.read_csv(file);
df = df.ix[df['number of reviews'] > 0]


grouped = df.groupby('number of reviews')
dat = grouped["average rating"].agg(np.mean)


file = 'amazon_data_full.csv'
df = pd.read_csv(file);
df = df.ix[df['number of reviews'] > 0]
df = df[df['salesrank']>0]


grouped = df.groupby('average rating')
grouped.size()
dat1 = grouped["salesrank"].agg(np.mean)
dat2 = grouped['number of reviews'].agg(np.mean)


figure = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(dat1,'.r',markersize = 15)
plt.xlabel('Product Rating',fontsize=22)
plt.ylabel('Mean Salesrank',fontsize=22)
plt.title('Mean Salesrank as a Fucntion of Product Rating',fontsize=22,y=1.04)
plt.xlim([0.75,5.25])
plt.xticks(np.arange(1,5.5,0.5))
plt.tight_layout()
plt.savefig("Fig5.pdf",format="pdf")
plt.show()


figure = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(dat2,'.r',markersize = 15)
plt.xlabel('Product Rating',fontsize=22)
plt.ylabel('Mean Number of Reviews',fontsize=22)
plt.title('Mean Number of Reviews as a Function of Product Rating',fontsize=22,y=1.04)
plt.xlim([0.75,5.25])
plt.ylim([0, 32])
plt.xticks(np.arange(1,5.5,0.5))
plt.tight_layout()
plt.savefig("Fig6.pdf",format="pdf")
plt.show()


grouped = df.groupby('salesrank')
dat = grouped["number of reviews"].agg(np.mean)
figure = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(dat,'b')
plt.xlabel('Salesrank', fontsize = 22)
plt.ylabel('Mean Number of reviews',fontsize = 22)
plt.title('Mean Number of Reviews as a Function of Salesrank',fontsize=22,y=1.04)
#plt.xticks(np.arange(1000000,5000000,1000000))
plt.ticklabel_format(style='sci', axis='x',scilimits=(0,0))
plt.tight_layout()
plt.savefig("Fig7.pdf",format="pdf")
plt.show()


file = 'amazon_data_full.csv'
df = pd.read_csv(file);
df = df.ix[df['number of reviews'] > 0]
#df = df[df['salesrank']>0]


categories = df.groupby('group')
avgs = categories["average rating"].agg(np.mean)
#avgs.sort_index().plot(kind='barh')
ax = avgs.sort_values(ascending=1).plot(kind = 'barh',figsize=[8,6])
plt.xlabel('Mean Rating',fontsize=22)
plt.ylabel('Product Group',fontsize=22)
plt.title('Mean Product Rating by Group',fontsize=22,y=1.04)
plt.xlim([0, 5.0])
y = np.array(avgs.sort_values(ascending=1))
y = np.around(y, decimals=2)
ind = np.arange(len(y))
for i, v in enumerate(y):
    ax.text(v-0.29, i-0.18 , str(v), color='white',fontsize=14)
plt.tight_layout()
plt.savefig("Fig8.pdf",format="pdf")
plt.show()


grouped = df.groupby(['group'])
A = grouped.agg(np.mean)
A = A.ix[A['number of reviews'] > 10]
ax = A['average rating'].sort_values(ascending=1).plot(kind = 'barh',figsize=[8,6],width = 0.5)
plt.xlabel('Mean Rating (Mininum 10 Reviews)',fontsize=22)
plt.ylabel('Product Group',fontsize=22)
plt.title('Mean Product Rating by Group (Min 10 Reviews)',fontsize=22,y=1.04)
plt.xlim([0, 5.0])
y = np.array(A['average rating'].sort_values(ascending=1))
y = np.around(y, decimals=2)
ind = np.arange(len(y))
for i, v in enumerate(y):
    ax.text(v-0.29, i-0.18 , str(v), color='white',fontsize=14)
plt.tight_layout()
plt.savefig("Fig9.pdf",format="pdf")
plt.show()




