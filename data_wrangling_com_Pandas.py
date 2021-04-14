#============================================================================
# Data Wrangling da base "Shakira/YouTube Spam Collection" usando a 'pandas'
#
# O BD contém uma coleção com 370 comentários sobre um vídeo da Shakira no YouTube
# https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
#
# 174 comentários são spam e 196 são normais
#============================================================================

#--------------------------------------------------------------------
#Passo 1: importar os "dados brutos" (raw data) e fazer verificação
#--------------------------------------------------------------------
import pandas as pd

nom_arq = 'https://raw.githubusercontent.com/edubd/CursoOutono/main/Youtube05-Shakira.csv'
bd = pd.read_csv(nom_arq)

print('* * IMPORTAÇÃO DO ARQUIVO: ', nom_arq)
print('-num. linhas: ', bd.shape[0])
print('-num. colunas: ', bd.shape[1])
print('-atributos: ', bd.columns)
print('-tipos dos atributos: ')
print(bd.dtypes)
print('-primeiras linhas: ')
print(bd.head())
print('-últimas linhas: ')
print(bd.tail())

#--------------------------------------------------------------------
#Passo 2: Data Wrangling - o objetivo é transformar 
#         em um BD com 4 colunas binárias:
# 
#         CLASS        : classe do comentário: 0=Normal; 1=Spam
#         LINK         : indica se CONTENT possui um link       
#         MUITAS_LETRAS: indica se CONTENT possui um número grande de letras        
#         MUITOS_MAIUSC: indica se a proporção de letras maiúsculas em CONTENT 
#                        é grande       
#--------------------------------------------------------------------

# * * 2.1 faz uma primeira transformação

#recebe como entrada um comentário (CONTENT) e retorna o seu número de caracteres
def f_total_letras(comentario):
    s = comentario.replace('\ufeff','').strip()
    return len(s)


#recebe como entrada um comentário (CONTENT) e retorna um valor entre 0 e 1 
#que indica a proporção de letras maiúsculas em relação ao total de caracteres 
def f_maiusc(comentario):
    s = comentario.replace('\ufeff','').strip()
    tot_letras = len([c for c in s if c != ' '])
    tot_maiusc = len([c for c in s if c >= 'A' and c <= 'Z'])
    
    return 0 if tot_letras == 0 else tot_maiusc / tot_letras


#recebe como entrada um comentário (CONTENT) e retorna 1 se ele possuir 
#algum link ou 0 caso contrário
def f_possui_link(comentario):
    s = comentario.replace('\ufeff','').strip()
    if (
        (s.find('http') > -1) or
        (s.find('www.') > -1) or
        (s.find('.com') > -1)
        ):
        return 1
    else:
        return 0

#aplica as funções para gerar o primeiro DataFrame transformado     
#com os atributos CONTENT, CLASS, TOT_LETRAS, MAIUSC e LINK        
D = pd.DataFrame(bd[['CONTENT','CLASS']])
D['TOT_LETRAS'] = D['CONTENT'].apply(f_total_letras)
D['MAIUSC'] = D['CONTENT'].apply(f_maiusc)
D['LINK'] = D['CONTENT'].apply(f_possui_link)


# * * 2.2 gera estatísticas sobre os novos atributos
i = 0
for nome_atributo in D.columns[2:5]:
    att = D[nome_atributo]
    att_dtype = att.dtype
    att_tam_dominio = att.unique().size
    att_tem_nulo = any(att.isnull())
    i += 1
    
    if (att_tam_dominio <= 2): 
        print("("+str(i)+") atributo:", nome_atributo, "\n",
              "dtype:", att_dtype, "\n\t",
              "nulos: ", att_tem_nulo, "\n\t",
              "freq. dos valores:")
        print(att.value_counts())
    else:
        print("("+str(i)+") atributo:", nome_atributo, "\n\t",
              "dtype:", att_dtype, "\n\t",
              "nulos: ", att_tem_nulo, "\n\t",
              "min: ", att.min(), "\n\t",
              "max: ", att.max(), "\n\t",
              "média: ", round(att.mean(),2), "\n\t",
              "mediana: ", round(att.median(),2), "\n\t",
              "d.p.: ", round(att.std(),2))


# * * 2.3 gera boxplots para os atributos TOT_LETRAS e MAIUSC
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
boxplot = D.boxplot(column=['TOT_LETRAS'], showmeans=True)

fig2, ax1 = plt.subplots()
boxplot = D.boxplot(column=['MAIUSC'], showmeans=True)


# * * 2.4 em função das estatísticas e dos boxplots, o código abaixo
#         gera a base de dados transformada final!


#f_muitas_letras: se palavra tiver mais de 200 letras retorna 1. 
#                 Caso contrário, retorna 0
#                 (eu cheguei ao valor 200 pelo IQR, analisando o boxplot)
def f_muitas_letras(tot_letras):
    return 0 if tot_letras <= 200 else 1


#f_muitso_maiusc: se mais de 25% das letras forem maiúsculas retorna 1. 
#                 Caso contrário, retorna 0
#                 (eu cheguei ao valor 25% pelo IQR, analisando o boxplot)
def f_muitos_maiusc(pct_letras):
    return 0 if pct_letras <= 0.25 else 1


D['MUITAS_LETRAS'] = D['TOT_LETRAS'].apply(f_muitas_letras)
D['MUITOS_MAIUSC'] = D['MAIUSC'].apply(f_muitos_maiusc)

#--------------------------------------------------------------------
#Passo 3: Usa a scikit-learn para gerar uma árvore de decisão capaz de 
#         representar a base transformada final de forma gráfica
#
#         Será que os 3 atributos derivados são suficientes para
#         identificar um comentário como spam?
#
#         Obs.: embora a árvore de decisão seja uma técnica de classificação
#               ela também pode ser usada em tarefas descritivas (ou seja, 
#               para descrever as características de uma base de dados)
#--------------------------------------------------------------------
from sklearn import tree

X = D.iloc[:,4:7]
Y = D.iloc[:,1]

arvore = tree.DecisionTreeClassifier()
arvore.fit(X,Y)

plt.figure(figsize=(12,12))
tree.plot_tree(arvore, fontsize=10, feature_names = list(D.columns[4:7]))
plt.show()
