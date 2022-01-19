import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
model = joblib.load('svm_model.sav')

pd.set_option("display.max_rows",None)
df = pd.read_csv('heart.csv')
df.ExerciseAngina = df.ExerciseAngina.replace(to_replace = ['Y','N'],value = [1,0])
df.Sex = df.Sex.replace(to_replace = ['F','M'],value = [1,0])
df.ChestPainType = df.ChestPainType.replace(to_replace = ['ATA','NAP','ASY','TA'], value = [1,2,3,4])
restingesg_mapping = {k:v for v, k in enumerate(df.RestingECG.unique())}
df['RestingECG'] = df['RestingECG'].map(restingesg_mapping)
ST_Slope = {k:v for v, k in enumerate(df.ST_Slope.unique())}
df['ST_Slope'] = df['ST_Slope'].map(ST_Slope)

X = df.drop('HeartDisease', axis=1)
Y = df[['HeartDisease']]

preds = model.predict(X)
print('Accuracy Score:', metrics.accuracy_score(Y, preds))

list = []
class Person:
    """
    A representation of a person
    Attributes:
        Firstname(string)
        Lastname(String)
        Health data(obj and int)
    """
    list = []
    def __init__(self, firstname, lastname,Age,Sex,ChestPainType, RestingBP,Cholesterol,FastingBS,
                 RestingECG,MaxHR,Exercising,Oldpeak,ST_Slope):
        self.firstname = firstname
        self.lastname = lastname
        self.Age = Age
        self.Sex = Sex
        self.ChestPainType = ChestPainType
        self.RestingBP = RestingBP
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.RestingECG = RestingECG
        self.MaxHR = MaxHR
        self.Exercising=Exercising
        self.Oldpeak = Oldpeak
        self.ST_slop = ST_Slope


    def show_prediction(self):

        list = [self.Age,self.Sex,self.ChestPainType, self.RestingBP,self.Cholesterol,
                self.FastingBS ,self.RestingECG, self.MaxHR,self.Exercising,self.Oldpeak,self.ST_slop]
        firstname = self.firstname
        lastname = self.lastname
        print(list)
        new_list = [int(item) for item in list]
        print( new_list)


        prediction = model.predict([new_list])
        prob = model.predict_proba([new_list])
        p_list = []
        for sublist in prob:
            for item in sublist:
                p_list.append(item)
        p = p_list[int(prediction)]
        return print("patient first name:" + firstname + " patient lastname:" +lastname+" Heartdisea: %2.3f"% int(prediction) + " percentage %2.3f"% p)

    @classmethod
    def get_user_input(self):
        while 1:
            try:
                firstname = input('Enter first name: ')
                lastname = input('Enter last name: ')
                Age = input('Enter Age: ')
                Sex = input('Enter Sex 1,2: ')#['F','M']
                ChestPainType = input('Enter ChestPainType 1,2,3,4: ')#['ATA','NAP','ASY','TA']
                RestingBP = input('Enter RestingBP: ')
                Cholesterol = input('Enter Cholesterol: ')
                FastingBS  = input('Enter FastingBS: ')
                RestingECG = input('Enter RestingECG 1,2: ')#[normal,st]
                MaxHR  = input('Enter MaxHR: ')
                Exercising = input('Enter Exercising 1,0: ')#Y,N
                Oldpeak = input('Enter Oldpeak: ')
                ST_Slope = input('Enter ST_slop1,2: ')#up,flat


                return self(firstname, lastname,Age,Sex,ChestPainType, RestingBP,Cholesterol,FastingBS,
                 RestingECG,MaxHR,Exercising,Oldpeak,ST_Slope)
            except:
                print('Invalid input!')
                continue


# creating a person object and returning their full name
person = Person.get_user_input()
person.show_prediction()

