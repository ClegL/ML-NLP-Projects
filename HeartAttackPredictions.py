#!/usr/bin/env python
# coding: utf-8
Ce modèle de prédiction est un projet que je fais pour me remettre dans le bain au niveau du machine learning après cette période de vacances, et aussi peut-être me donner des idées pour un autre projet que j'ai entête depuis un moment. J'ai déjà commencé avec un petit dashboard sur les animes, et je poursuis à présent avec un peu de machine learning.

Pour ce projet j'ai utilisé le dataset "❤️ Heart Attack Risk Factors Dataset" par Waqar Ali disponible sur Kaggle.

---

This prediction model on heart attacks is just a little project I'm doing to get back into machine learning after this holiday period, and maybe give me other idea for another project that's been brewing in my mind for a while. I've already started with a small dashboard on anime, and I'm now continuing with a bit of machine learning.

For this project I used the dataset "❤️ Heart Attack Risk Factors Dataset" by Waqar Ali available on Kaggle.
# In[1]:


import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Ulysse\\Desktop\\Projets persos + Dashboards\\Datasets\\heart_attack_dataset.csv")

shape_df = len(df)
print("Shape df :", shape_df)

df.head()

Quick check just to not get any surprises during the training and all that (null values, uniques values).

Check rapide pour ne pas avoir de surprises durant l'entraînement (notamment valeurs nulles et uniques).
# In[2]:


df.isnull().sum()


# In[3]:


df['Treatment'].nunique()


# In[4]:


df['Chest Pain Type'].nunique()


# In[5]:


df.info()


# In[6]:


df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Blood Pressure (mmHg)'] = pd.to_numeric(df['Blood Pressure (mmHg)'], errors='coerce')
df['Cholesterol (mg/dL)'] = pd.to_numeric(df['Blood Pressure (mmHg)'], errors = 'coerce')

# Juste pour être sûr que ça ne me créera pas de problèmes comme j'ai pu en avoir sur d'autres projets par manque de vigilance,
# Je convertis les colonnes en numérique et m'occupent des erreurs potentielles.

# Just to make sure it won't cause me problems like I've had on other projects due to lack of vigilance,
# I convert the columns to numeric and take care of potential errors.

df.head()


# In[7]:


sns.regplot(x = 'Blood Pressure (mmHg)',
            y = 'Age',
            data = df)
plt.show()

Comme on peut le voir, il n'ya aucune tendance nette entre l'âge et la pression artérielle, en tous cas pas dans ces données.

As we can see, there is no clear trend between age and blood pressure, at least not in these data.
# In[8]:


sns.regplot(x = 'Blood Pressure (mmHg)',
            y = 'Cholesterol (mg/dL)',
            data = df)
plt.show()

Comme prévu (et comme nous le savons tous, je pense) le graphique ci-dessus montre une forte corrélation positive entre la pression artérielle et le cholestérol dans cet ensemble de données, ce qui peut aider à identifier des patients à risque de complications cardiovasculaires en fonction de ces deux variables.

Ça fait sens.

D'abord il me faut encoder mes données pour m'en servir. Je vais donc utiliser le label encoder pour mes colonnes à chaînes de caractères.

---

As expected (and as I think we all know) the graph above shows a strong positive correlation between blood pressure and cholesterol in this dataset, which can help identify patients at risk of cardiovascular complications based on these two variables.

Makes sense.

First I need to encode my data to use it. So I'm going to use the encoder label for my string columns.
# In[9]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Has Diabetes'] = label_encoder.fit_transform(df['Has Diabetes'])
df['Smoking Status'] = label_encoder.fit_transform(df['Smoking Status'])
df['Chest Pain Type'] = label_encoder.fit_transform(df['Chest Pain Type'])
df['Treatment'] = label_encoder.fit_transform(df['Treatment'])

df

Il me fallait également une colonne cible 'Heart Attack' simulée pour pouvoir faire la prédiction, et pour cela je me suis basé sur plusieurs critères :

- Le Type de traitement reçu : les personnes ayant subi une angioplastie ou une greffe de pontage coronaire (Coronary Artery Bypass Graft) pourraient être considérées comme ayant eu une crise cardiaque.
- Type de douleur thoracique : les personnes ayant des douleurs thoraciques typiques ou atypiques (angine) sont peut-être plus susceptibles d'avoir eu une crise cardiaque.
- Facteurs de risque : pression artérielle élevée, cholestérol élevé, diabète, tabagisme ...

---

I also needed a target simulated column 'Heart Attack' to be able to make the prediction, and for this I based it on several criteria:

- Type of treatment received: people who have had angioplasty or a coronary artery bypass graft could be considered to have had a heart attack.
- Type of chest pain: people with typical or atypical chest pain (angina) may be more likely to have had a heart attack.
- Risk factors: high blood pressure, high cholesterol, diabetes, smoking...
# In[10]:


df['Heart Attack'] = ((df['Age'] > 50) & 
                      (df['Cholesterol (mg/dL)'] > 240) & 
                      (df['Blood Pressure (mmHg)'] > 140)).astype(int)


# In[11]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split

X = df.drop(columns=['Heart Attack'])
y = df['Heart Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Je sépare toujours mon ensemble de données en 80/20, habitude de ma formation
# I always split my dataset in 80/20, training habit.

Je vais essayer plusieurs modèles et garderait celui qui a le meilleur résultat.
# In[13]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

100% d'accuracy, le modèle est surajusté.

100% Accuracy, overfitted.
# In[14]:


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

Pas de gros déséquilibre entre les classes. On va donc check le rapport de classification.

---

No big imbalance between classes. So we'll check the classification report.
# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=5)

print(f"Scores de validation croisée: {scores}")
print(f"Précision moyenne: {scores.mean() * 100:.2f}%")


# In[16]:


model = RandomForestClassifier(max_depth=5, n_estimators=50)

scores = cross_val_score(model, X, y, cv=5)

print(f"Scores de validation croisée: {scores}")
print(f"Précision moyenne: {scores.mean() * 100:.2f}%")


# In[17]:


print(df['Heart Attack'].value_counts())

D'après la heatmap mes corrélations sont relativement faibles entre mes caractéristiques et ma cible, celles qui ont des corrélations ont des corrélations bien trop faibles, comme Treatment_Coronary Artery Bypass Graft (CABG) et Heart Attack par exemple. Il y a également les colonnes 'Chest Pain Type' qui créé des complications, notamment en introduisant une multicolinéarité.

Je vais donc d'abord essayer d'autres modèles avant de voir si la réduction de la multicolinéarité est ma seule solution.

# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

print(classification_report(y_test, y_pred_rf))

Ok, pas bon du tout. Le modèle étant surajusté également, mon instinct me dit de m'occuper de la multicolinéarité alors c'est ce que je vais faire.
# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)


# In[20]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[21]:


X = X.drop(columns=['Cholesterol (mg/dL)'])


# In[22]:


vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)


# In[23]:


X = X.drop(columns=['Blood Pressure (mmHg)'])


# In[24]:


vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

print(classification_report(y_test, y_pred_rf))


# In[26]:


df

Je pense avoir saisi le problème : manque de données(le dataset n'a que 1000 lignes) + le fait que ma colonne cible est remplie de 0, ce qui encourage le surapprentissage. Juste pouur essayer, je vais essayer d'introduire des données synthétiques avec SMOTE avant d'arrêter ce projet et passer sur autre chose. 

I think I understand the issues: lack of data (the dataset has only 1000 rows) + the fact that my target column is filled with 0, which encourages overfitting. Just to try, I'll try to introduce synthetic data with SMOTE before stopping this project and moving on to something else.
# In[31]:


new_rows = pd.DataFrame({
    'Gender': [1, 0],
    'Age': [70, 65],
    'Blood Pressure (mmHg)': [180, 160],
    'Cholesterol (mg/dL)': [240, 220],
    'Has Diabetes': [1, 1],
    'Smoking Status': [2, 2],
    'Chest Pain Type': [3, 2],
    'Treatment': [1, 3],
    'Heart Attack': [1, 1]  
})

df = pd.concat([df, new_rows], ignore_index=True)

print(df['Heart Attack'].value_counts()) 


# In[32]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Heart Attack'])
y = df['Heart Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Taille avant SMOTE: {X_train.shape}")
print(f"Taille après SMOTE: {X_train_smote.shape}")
print(f"Distribution après SMOTE dans y_train_smote:\n{y_train_smote.value_counts()}")


# In[33]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Heart Attack'])
y = df['Heart Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42, k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Taille avant SMOTE: {X_train.shape}")
print(f"Taille après SMOTE: {X_train_smote.shape}")
print(f"Distribution après SMOTE dans y_train_smote:\n{y_train_smote.value_counts()}")


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[35]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[37]:




Je pense m'arrêter ici. 

L'objectif initial de ce projet était de construire un modèle de prédiction des crises cardiaques à partir de facteurs de risques médicaux tels que l'âge, la pression artérielle, le taux de cholestérol, le statut de diabète, le type de douleur thoracique, et d'autres variables pertinentes. Bien que le projet ait permis de développer et d'entraîner un modèle de machine learning (Random Forest) pour cette tâche, plusieurs limitations liées à la nature des données ont été identifiées.

Le faible nombre d'occurrences de la classe positive (crise cardiaque) a empêché le modèle d'apprendre correctement à les identifier. Des techniques comme le sur-échantillonnage (SMOTE) ont été explorées, mais elles se sont heurtées aux limites du faible nombre de cas positifs dans le jeu de données.

L'analyse de la variance d'inflation (VIF) a montré que certaines variables étaient colinéaires, ce qui a pu affecter la stabilité des coefficients du modèle.

Un nombre plus important de données, notamment des cas de crises cardiaques, serait nécessaire pour entraîner un modèle plus robuste et obtenir des résultats fiables.

Ce projet a permis de me remettre dans le bain avec la préparation des données, le traitement des variables catégoriques et la construction de modèles de machine learning dans le contexte des données déséquilibrées. Cependant, les résultats obtenus sont largement limités par la qualité et la diversité des données. Un travail futur avec des données plus riches permettrait de construire un modèle prédictif réellement utile pour évaluer le risque de crise cardiaque.

---

I think I'll stop here. 

The initial goal of this project was to build a heart attack prediction model based on medical risk factors such as age, blood pressure, cholesterol level, diabetes status, type of chest pain, and other relevant variables. Although the project allowed to develop and train a machine learning model (Random Forest) for this task, several limitations related to the nature of the data were identified.

The low number of occurrences of the positive class (heart attack) prevented the model from learning to correctly identify them. Techniques such as oversampling (SMOTE) were explored, but they ran into the limitations of the low number of positive cases in the dataset.

Variance inflation analysis (VIF) showed that some variables were collinear, which could have affected the stability of the model coefficients.

A larger number of data, especially heart attack cases, would be necessary to train a more robust model and obtain reliable results.

This project allowed me to get back into the swing of things with data preparation, categorical variable processing, and building machine learning models in the context of imbalanced data. However, the results obtained are largely limited by the quality and diversity of the data. Future work with richer data would allow building a predictive model that would be truly useful for assessing heart attack risk.