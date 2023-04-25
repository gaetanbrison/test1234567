import streamlit as st
import pandas as pd
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt

def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()
df = pd.read_csv("hm2.csv",error_bad_lines=False,sep=";")

app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])

if app_mode == 'Summary 🚀':
    st.subheader("01 Summary Page🚀")



    #df3 = pd.read_csv("hmnew.csv")


    ### The st.title function sets the title of the web application to "Midterm Project - 01 Introduction Page".
    st.title("Midterm Project - 01 Introduction Page")

    ### The first two lines of the code load an image and display it using the st.image function.
    image_logo = Image.open('images/logo.png')
    st.image(image_logo, width=100)

    ### The st.subheader function sets the title of the web application to "H&M Sales and Customer Analysis".
    st.subheader("H&M Sales and Customer Analysis")
    st.subheader("Analyzing and Forecasting Future Sells at H&M")

    ### The st.number_input function creates a widget that allows the user to input a number. The st.radio function creates a radio button widget that allows the user to select either "Head" or "Tail".
    num = st.number_input('No. of Rows', 5, 100)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
    ### the st.dataframe function displays the data frame.
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    ### The st.markdown function is used to display some text and headings on the web application.
    st.markdown("### 01 - Show  Dataset")

    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    ### The st.text and st.write functions display the shape of the data frame and some information about the variables in the data set.
    st.text('(Rows,Columns)')
    st.write(df.shape)

    st.markdown("##### variables ➡️")
    st.markdown(" **customer ID**: quantitative identification for each customer")
    st.markdown(" **article ID**: quantitative identification for each article of clothing")
    st.markdown(" **price**: price of article (multipy by 1000 for USD)")
    st.markdown(" **product code**: quantitative identification for each product")
    st.markdown(" **product name**: name identification for each product")
    st.markdown(" **product type name**: qualitative identification for each product")
    st.markdown(" **product group name**: qualitative grouping for each product")
    st.markdown(" **graphical appearance No**: quantitative identification for pattern for each product")
    st.markdown(" **graphical appearance name**: qualitative identification for pattern for each product")
    st.markdown(" **color group code**: quantitative identification for color for each product")
    st.markdown(" **color group name**: qualitative identification for color for each product")
    st.markdown(" **perceieved color value ID**: quantitative identification for perceived color for each product")
    st.markdown(" **perceieved color value name**: qualitative identification for perceived color for each product")
    st.markdown(" **department No**: quantitative identification for department category for each product")
    st.markdown(" **department name**: qualitative identification for department category for each product")
    st.markdown(" **index group No**: quantitative identification for index group for each product")
    st.markdown(" **index group name**: qualitative identification for index group for each product")
    st.markdown(" **section No**: quantitative identification for section category for each product")
    st.markdown(" **section name**: qualitative identification for section category for each product")
    st.markdown(" **garment group No**: quantitative identification for garment group for each product")
    st.markdown(" **garment group name**: qualitative identification for garment group for each product")
    st.markdown(" **detail description**: detailed description for each product")


    st.dataframe(df.head(3))


    ### The st.markdown and st.dataframe functions display the descriptive statistics of the data set.
    st.markdown("### 02 - Description")
    st.dataframe(df.describe())


    ### The st.markdown, st.write, and st.warning functions are used to display information about the missing values in the data set.
    st.markdown("### 03 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")


    st.markdown("### 04 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")    
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")


if app_mode == 'Visualization 📊':
    st.subheader("02 Visualization Page📊")

    st.markdown("[![Foo](https://i.postimg.cc/GmG9XZ3b/Screenshot-2023-04-04-at-15-48-36.png)](https://lookerstudio.google.com/reporting/64a82b63-f5ec-4323-a5fb-c11a165805c3)")

if app_mode == 'Prediction 📈':




### The st.title() function sets the title of the Streamlit application to "Midterm Project - 03 Prediction Page".
    st.title("Midterm Project - 03 Prediction Page")

    ### The pd.read_csv() function loads a CSV file of H&M data into a Pandas DataFrame called "df".
    df = pd.read_csv("hmvis.csv")


    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    list_variables = df.columns
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    output_multi = st.multiselect("Select Explanatory Variables", list_var,default=["product_type_no","graphical_appearance_no","department_no"])

    new_df2 = new_df[output_multi]
    x =  new_df2
    y = df[select_variable]

    ### The train_test_split() function splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)

    ### The LinearRegression() function creates a linear regression model.
    lm = LinearRegression()

    ### The lm.fit() function fits the linear regression model to the training data.
    lm.fit(X_train,y_train)

    ###The lm.predict() function generates predictions for the testing data.
    predictions = lm.predict(X_test)

    ### The st.columns() function creates two columns to display the feature columns and target column.
    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(x.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    ### The st.subheader() function creates a subheading for the results section.
    st.subheader('🎯 Results')

    ### The st.write() function displays various metrics for the linear regression model, including the variance explained, mean absolute error, mean squared error, and R-squared score. The results are rounded to two decimal places using the np.round() function.
    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))