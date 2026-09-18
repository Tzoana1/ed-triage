import pandas as pd 
import numpy as np
#Φόρτωση του αρχείου 
df=pd.read_excel("1stproject-TestSet.csv")
print("Total rows loaded:",len(df))
#Μετατροπή βασικής μεταβλητής σε 0 και 1 
df['apotelesma']=np.where(df['disposition'].str.contains('Admit', case=False, na=False),1,0)
#Επιλογή των στηλών για το μοντέλο
keep_columns=['esi','age','gender','n_edvisits','n_admissions','arrivalmode','chestpain','abdomnlpain','uti','apotelesma']
df=df[keep_columns].copy()
#Καθαρισμός κενών τιμών
df=df.dropna()
#Χωρισμός στοιχείων από το αποτέλεσμα 
xaraktiristika=pd.get_dummies(df.drop(['apotelesma'],axis=1), drop_first=True)
apotelesma=df['apotelesma']
print("Έτοιμο! Σύνολο:", len(xaraktiristika))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
# Χωρισμός σε train και test 
xar_train,xar_test,apot_train,apot_test=train_test_split(xaraktiristika,apotelesma,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
xar_train=scaler.fit_transform(xar_train)
xar_test=scaler.transform(xar_test)
# Δημιουργία του μηχανισμού Stacking 
# Ορισμός 3 διαφορετικών μοντέλων 
base_models= [('m1',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
              ('m2',RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)), 
              ('m3',KNeighborsClassifier(n_neighbors=7))]
#Τελικό μοντέλο που ενώνει τις προβλέψεις 
m_final=LogisticRegression(class_weight='balanced',random_state=42)
#Δημιουργία τελικού μηχανισμού Stacking
teliko=StackingClassifier(estimators=base_models,final_estimator=m_final,cv=5)
# Μήνυμα αναμονής 
print("\nΥπολογισμός...Αναμονή...")
teliko.fit(xar_train, apot_train)

problepseis= teliko.predict(xar_test)
#Αποτελέσματα 
print("\nΑΠΟΤΕΛΕΣΜΑΤΑ")
print("ΕΠΙΤΥΧΙΑ:", accuracy_score(apot_test,problepseis)*100)
print("\nΠΙΝΑΚΑΣ:")
print(classification_report(apot_test,problepseis))

