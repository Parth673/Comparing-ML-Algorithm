import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, hamming_loss
from sklearn.preprocessing import LabelEncoder
import time
# ML Model------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

iris = sns.load_dataset('iris')
#fig3D = px.scatter_3d(iris, x = 'sepal_length', y = 'sepal_width', z = 'petal_length', color = 'species')
#fig3D.show()
x = iris.iloc[:, :2]
y = iris.iloc[:,-1]
le = LabelEncoder()
y_trans = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x,y_trans, test_size=0.3, random_state=32)

def draw_meshgrid():
    a = np.arange(start=x_train.iloc[:, 0].min() - 1, stop=x_train.iloc[:, 0].max() + 1, step=0.01)
    b = np.arange(start=x_train.iloc[:, 1].min() - 1, stop=x_train.iloc[:, 1].max() + 1, step=0.01)
    xx, yy = np.meshgrid(a, b)
    in_arr = np.array([xx.ravel(), yy.ravel()]).T
    return xx, yy, in_arr

plt.style.use('seaborn-v0_8-darkgrid')

#Hyper Parameter----------------------------------
st.sidebar.markdown("# Logistic Regression Classifier")
c_input =  float(st.sidebar.number_input('c_input',value=0.5))
max_iter = int(st.sidebar.slider('max_iter', min_value=0, max_value=1000))
penalty = st.sidebar.selectbox('Regularization', ('l2', 'l1','elasticnet','none'))
l1_ratio = float(st.sidebar.number_input('l1_ratio',value=0.5))

#Plot-----------------------------------------------
# Load initial graph
fig, ax = plt.subplots()
# Plot initial graph
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train, cmap='rainbow')
orig = st.pyplot(fig)

#Button OnClick-------------------------------------
if st.sidebar.button('Run Algorithm'):
    orig.empty()

    start = time.time()
    lor = LogisticRegression(penalty='elasticnet', C=c_input, solver='saga', max_iter=max_iter, multi_class='multinomial',l1_ratio=l1_ratio)
    lor.fit(x_train, y_train)
    y_pred = lor.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    ham_los_val = hamming_loss(y_test, y_pred)

    xx, yy, in_arr = draw_meshgrid()
    lable = lor.predict(in_arr)
    end = time.time()
    # Put the result into a color plot
    plt.figure(1, figsize=(6, 5))
    plt.pcolormesh(xx, yy, lable.reshape(xx.shape), cmap='rainbow', alpha=0.4)

    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train, cmap='rainbow', alpha=0.8)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
    orig = st.pyplot(fig)
    # Accuracy & Time
    #st.text('Time: ',  time.time() - start)
    st.markdown(':red[Time:] ' +  str(end - start))
    st.markdown(':red[Accuracy:] ' + str(accuracy))
    st.markdown(':red[Hamming_loss:] ' + str(ham_los_val))

