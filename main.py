#!/usr/bin/env python
# coding: utf-8

# # Data-Science-0 - Semana 2 - Pré-processamento de dados em Python
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Leitura do CSV
black_friday = pd.read_csv("black_friday.csv")


# ## Inicio da análise

# In[3]:


#Listagem das 5 primeiras linhas do dataframe
black_friday.head()


# In[4]:


#Informações do dataframe
black_friday.info()


# In[5]:


#Descrição das variaveis do Dataframe
black_friday.describe()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[6]:


#Listagem do numero de linhas e colunas do Dataset
def q1():
    return black_friday.shape

q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[7]:


#Filtro de colunas usuario, sexo e idade
dataframe = black_friday[['User_ID','Gender','Age']]
dataframe.head()


# In[10]:


#Checando se a Idade está dentro do range
dataframe = dataframe[dataframe.Age == '26-35']
dataframe.head()


# In[11]:


#Filtro de mulheres dentro do range
dataframe = dataframe[dataframe.Gender == 'F']
dataframe.head()


# In[12]:


#batendo a chave de usuário, unico registro
dataframe['User_ID'].nunique()


# In[14]:


#Função que faz a triagem de usuario, gexo e idade
def q2():
    dataframe = black_friday[['User_ID','Gender','Age']]
    dataframe = dataframe[dataframe.Age == '26-35']
    dataframe = dataframe[dataframe.Gender == 'F']
    return dataframe['User_ID'].nunique()
q2()


# In[15]:


# Resposta oficial, sem considerar os usuários únicos:
black_friday.query('Gender == "F" & Age == "26-35"').shape[0] 


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[16]:


#Listagem de usuários unicos do Dataset
black_friday['User_ID'].unique()


# In[17]:


#Total de usuarios unicos do Dataset
def q3():
    return black_friday['User_ID'].nunique()

q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[18]:


#Tipos de variaveis diferentes dos Dataset
black_friday.dtypes


# In[19]:


#Função que retorna números de variaveis diferentes do Dataset
def q4():
    return black_friday.dtypes.nunique()
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `NaN` etc)? Responda como um único escalar entre 0 e 1.

# In[20]:


#qual linha tem pelo menos um NA? #total de linhas
total_linhas = black_friday.shape[0]


# In[21]:


total_linhas_sem_na = black_friday.dropna(axis='index').shape[0]


# In[22]:


total_linhas_na = total_linhas - total_linhas_sem_na


# In[23]:


#Total da % de linhas que possui ao menos um valor Null no Dataset
total_linhas_na/total_linhas


# In[24]:


def q5():
    total_linhas = black_friday.shape[0]
    total_linhas_sem_na = black_friday.dropna(axis='index').shape[0]
    total_linhas_na = total_linhas - total_linhas_sem_na
    return total_linhas_na/total_linhas

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[45]:


#Total da Soma de valores Null 
black_friday.isna().sum()


# In[46]:


#Total da Soma de valores Null com Max
black_friday.isna().sum().max()


# In[49]:


#Função que lista  total da Soma de valores Null com Max
def q6():
    return int(black_friday.isna().sum().max())

q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[28]:


#Analisando os dados da Product_Category_3  para saber qual o valor mais frequente aparece no Dataset
black_friday['Product_Category_3'].head()


# In[29]:


black_friday['Product_Category_3'].dropna().mode()


# In[30]:


def q7():
    return float(black_friday['Product_Category_3'].dropna().mode())

q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# $$ norm = (x - min) / (max - min) $$

# In[31]:


min = black_friday.Purchase.min()
max = black_friday.Purchase.max()


# In[51]:


(black_friday.Purchase - min) / (max-min)


# In[52]:


def q8():
    min = black_friday.Purchase.min()
    max = black_friday.Purchase.max()
    return float(((black_friday.Purchase - min) / (max-min)).mean())

q8()  


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# $$ pad = (x - \bar{x}) / s $$

# In[34]:


media = black_friday.Purchase.mean()
desvio = black_friday.Purchase.std()


# In[35]:


padrao = pd.DataFrame(black_friday['Purchase'].copy())


# In[36]:


padrao.Purchase = (padrao.Purchase - media) / desvio
padrao


# In[37]:


padrao.Purchase = padrao[padrao.Purchase <= 1]
padrao.Purchase = padrao[padrao.Purchase >= -1]


# In[38]:


padrao.dropna()


# In[39]:


padrao.shape[0]


# In[40]:


def q9():
    compras = black_friday['Purchase']
    padrao = (compras - compras.mean()) / compras.std()
    return int(((padrao >= -1) & (padrao <= 1)).sum())

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[41]:


question_ten = black_friday[['Product_Category_2','Product_Category_3']]


# In[42]:


nulos = question_ten[question_ten['Product_Category_2'].isna()]
nulos


# In[43]:


def q10():
    return black_friday['Product_Category_2'].isna().equals(black_friday['Product_Category_2'].isna())

q10()


# In[ ]:




