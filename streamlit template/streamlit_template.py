#----------- Welcome to the Streamlit template provided by So you want to be a data scientist? -----------#

#	In this template, we will:
# 		* go over the commonly used Streamlit structures
#		* learn how to collect input from the user
#		* see how to import data
# 		* learn how to customize the app
#		* have some examples on visualizing the data

# FIRST: Run the app, interact with it, and then come back and go through the code

# TO RUN THE APP:
#	* open terminal
#	* navigate to streamlit file location
#	* to install required libraries, run: "pip install -r requirements.txt"
#	* use command: "streamlit run streamlit_template.py"


# Need more help understanding and setting up your Streamlit app? Check out my tutorial on YouTube: https://www.youtube.com/watch?v=-IM3531b1XU&list=PLM8lYG2MzHmTATqBUZCQW9w816ndU0xHc
# You can get more information on everything on the Streamlit documentation: https://docs.streamlit.io/en/stable/api.html


# The data being used in this app is a truncated version of the data that you can download here: https://s3.amazonaws.com/tripdata/index.html
# more info about the data can be found here: https://www.citibikenyc.com/system-data


# I've added ToDos for you to interact with the template and get familiar with it
# Search for the phrase "TODO" and you'll see them


# 1 --- first and foremost, we import the necessary libraries
import pandas as pd
import streamlit as st

import plotly.express as px
#######################################





# 2 --- you can add some css to your Streamlit app to customize it
# TODO: Change values below and observer the changes in your app
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
#######################################






# 3 --- build the structure of your app


# Streamlit apps can be split into sections


# container -> horizontal sections
# columns -> vertical sections (can be created inside containers or directly in the app)
# sidebar -> a vertical bar on the side of your app


# here is how to create containers
header_container = st.beta_container()
stats_container = st.beta_container()	
#######################################



# You can place things (titles, images, text, plots, dataframes, columns etc.) inside a container
with header_container:

	# for example a logo or a image that looks like a website header
	st.image('logo.png')

	# different levels of text you can include in your app
	st.title("A cool new Streamlit app")
	st.header("Welcome!")
	st.subheader("This is a great app")
	st.write("check it for yourself, if you don't believe me")








# Another container
with stats_container:





	# 4 --- You import datasets like you always do with pandas
	# 		if you'd like to import data from a database, you need to set up a database connection
	data = pd.read_csv('JC-202103-citibike-tripdata.csv')







	# 5 --- You can work with data, change it and filter it as you always do using Pandas or any other library
	start_station_list = ['All'] + data['start station name'].unique().tolist()
	end_station_list = ['All'] + data['end station name'].unique().tolist()








	# 6 --- collecting input from the user
	#		Steamlit has built in components to collect input from users


	# collect input using free text
	# the input of the user will be saved to the variable called "text_input"
	text_input = st.text_input("You can collect free text input from the user", 'Something')


	# collect input usinga list of options in a drop down format
	# TODO: change the option list to end_station_list and see what happens
	st.write('Or you can ask the user to select an option from the dropdown menu')
	s_station = st.selectbox('Which start station would you like to see?', start_station_list, key='start_station')

	# display the collected input
	st.write('You selected the station: ' + str(s_station))

	# you can filter/alter the data based on user input and display the results in a plot
	st.write('And display things based on what the user has selected')
	if s_station != 'All':
		display_data = data[data['start station name'] == s_station]

	else:
		display_data = data.copy()


	# display the dataset in a table format
	# if you'd like to customize it more, consider plotly tables: https://plotly.com/python/table/
	# I have a YouTube tutorial that can help you in this: https://youtu.be/CYi0pPWQ1Do
	st.write(display_data)


	# here is a different way of collecting data, namely multiple selection
	st.write('It is possible to have multiple selections too.')
	multi_select = st.multiselect('Which start station would you like to see?',start_station_list, key='start_station', default=['Harborside','Marin Light Rail'])


	# and yet another way, sliders
	# I get the range to be displayed from the tripduration column of the dataset
	# it is in seconds so I divide it by 3600 to get hours
	# the last parameter in the slider function is the default value
	slider_input = st.slider('How long should the trip be?', int(data['tripduration'].min()/3600), int(data['tripduration'].max()/3600), 25)






	# 7 --- creating columns inside a container 
	#		(you can create more than 2)
	bar_col, pie_col = st.beta_columns(2)

	# in order to display things inside columns, replace the st. with the column name when creating components

	# preparing data to display on pie chart
	user_type = data['usertype'].value_counts().reset_index()
	user_type.columns = ['user type','count']






	# 8 --- Creating plots and charts
	#		The simplest way is to use the built in Streamlit plots
	#		You can also use other plotting libraries with Streamlit
	#		The pie chart below is an example of using Plotly


	# preparing data to display in a bar chart
	start_location = data['start station name'].value_counts()

	# don't forget to add titles to your plots
	bar_col.subheader('Trip duration in minutes per start station')

	# This is the way to make a very simple bar chart 
	# Visit https://docs.streamlit.io/en/stable/api.html for more information on what other plots and charts are possible
	bar_col.bar_chart(start_location)





	# don't forget to add titles to your plots
	pie_col.subheader('How many of the users were subscribed?')

	# This is an example of a plotly pie chart
	fig = px.pie(user_type, values='count', names = 'user type', hover_name='user type')

	# TODO: change the values of the update_layout function and see the effect
	fig.update_layout(showlegend=False,
		width=400,
		height=400,
		margin=dict(l=1,r=1,b=1,t=1),
		font=dict(color='#383635', size=15))

	# this function adds labels to the pie chart
	# for more information on this chart, visit: https://plotly.com/python/pie-charts/
	fig.update_traces(textposition='inside', textinfo='percent+label')

	# after creating the chart, we display it on the app's screen using this command
	pie_col.write(fig)







