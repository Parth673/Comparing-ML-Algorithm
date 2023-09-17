import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import seaborn as sns

iris = sns.load_dataset('iris')

st.header('Algo Comparison')
st.subheader('by A018 :blue[Parth] :sunglasses:')

fig = px.scatter_3d(iris, x = 'petal_length', y = 'sepal_length', z = 'sepal_width', color = 'species')
#sepal_length, sepal_width, petal_length
st.plotly_chart(fig, use_container_width=True)


# 🏠_Ghar.py