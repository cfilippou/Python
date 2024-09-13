#Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 20)

#-------------------------------------------------------------------------------
# pizzas.csv overview
#-------------------------------------------------------------------------------
pizzas = pd.read_csv('pizzas.csv')
print(pizzas.info())
print(pizzas.head())

#cast 'object' dtype to string
objs = [col for col in pizzas.columns if pizzas[col].dtype == 'object']
pizzas[objs] = pizzas[objs].astype('string')

#-------------------------------------------------------------------------------
#pizza_types.csv overview
#-------------------------------------------------------------------------------
pizza_types = pd.read_csv('pizza_types.csv')
print(pizza_types.info())
print(pizza_types.head())

#cast 'object' dtype to string
objs = [col for col in pizza_types.columns if pizza_types[col].dtype == 'object']
pizza_types[objs] = pizza_types[objs].astype('string')

#-------------------------------------------------------------------------------
#orders.csv overview
#-------------------------------------------------------------------------------
orders = pd.read_csv('orders.csv', index_col='order_id')
print(orders.info())
print(orders.head())

#cast 'object' dtype to string
objs = [col for col in orders.columns if orders[col].dtype == 'object']
orders[objs] = orders[objs].astype('string')

#-------------------------------------------------------------------------------
#order_details.csv overview
#-------------------------------------------------------------------------------
order_details = pd.read_csv('order_details.csv')
print(order_details.info())
print(order_details.head())

#cast 'object' dtype to string
objs = [col for col in order_details.columns if order_details[col].dtype == 'object']
order_details[objs] = order_details[objs].astype('string')

#-------------------------------------------------------------------------------
#customers.csv overview
#-------------------------------------------------------------------------------
customers = pd.read_csv('customers.csv')
print(customers.info())
print(customers.head())

#cast 'object' dtype to string
objs = [col for col in customers.columns if customers[col].dtype == 'object']
customers[objs] = customers[objs].astype('string')

#The overviews shown us that there are no na values in any dataframe.
#So, functions like dropna, isna fillna etc, will not be used in this project.

#After reading all csv files, we have to merge these dataframes to one. 
#It is going to be done in several steps, till all dataframes are merged to one.

#-------------------------------------------------------------------------------
#STEP 1
#-------------------------------------------------------------------------------
#First dataframe is 'order_details' that contains all infromation about each order. This dataframe is merged with pizzas dataframe.
#The new dataframe is named 'all_info'
all_info = order_details.merge(pizzas, on='pizza_id')

#-------------------------------------------------------------------------------
#STEP 2
#-------------------------------------------------------------------------------
#Now, total amount of each order part must be calculated
all_info['total_amount'] = all_info['price'] * all_info['quantity']

#-------------------------------------------------------------------------------
#STEP 3
#-------------------------------------------------------------------------------
#Merge with pizza_types dataframe.
all_info = all_info.merge(pizza_types, on='pizza_type_id')

#-------------------------------------------------------------------------------
#STEP 4
#-------------------------------------------------------------------------------
#Merge with orders dataframe.
all_info = all_info.merge(orders, how='left', on='order_id')

#-------------------------------------------------------------------------------
#STEP 5
#-------------------------------------------------------------------------------
#Merge with customers dataframe. This time used more parameters because the index columns about
#customer_id have different names.
all_info = all_info.merge(customers, how='left', left_on='cust_id', right_on='id')
print(all_info)

#-------------------------------------------------------------------------------
#QUESTIONS AND GRAPHS PART
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#QUESTION 1. Which pizza size is most preferred by men and women? Show the result
# in a graph with seaborn
#-------------------------------------------------------------------------------
res_1 = all_info.groupby(['sex', 'size'])['pizza_id'].agg(['count'])
sns.barplot(data=res_1, x='sex', y='count', hue='size')
#The graph shows us that both men and women prefer the Large size of pizzas
#most of the times

#-------------------------------------------------------------------------------
#QUESTION 2. Create a new dataframe that contains id, full name, sex, total orders,
# total paid amound.
#-------------------------------------------------------------------------------
#Group by by 'cust_id' and calculate sum of 'total_amount' and how many times
#every cust_id is saved in the file.
res_2 = all_info.groupby('cust_id')['total_amount'].agg(['sum', 'count'])
#Merge with exact columns of all_info dataframe to get the full name of each customer.
res_2 = res_2.merge(all_info[['cust_id', 'full_name', 'sex']], how='left', on='cust_id')
#Remove duplicates
res_2 = res_2.drop_duplicates()
#Reorder columns
res_2 = res_2[['cust_id', 'full_name', 'sex', 'sum', 'count']]
#Rename columns
res_2.rename(columns={'sum':'total_paid_amount', 'count': 'num_of_orders'}, inplace=True)
#print result
print(res_2)

#-------------------------------------------------------------------------------
#QUESTION 3. Do men or women order more? Show the results in a barplot.
#-------------------------------------------------------------------------------
# The result of the previous question will be used.
res_3 = res_2.groupby('sex')['num_of_orders'].agg(['sum'])
res_3.apply(lambda x: 'Men' if x['F'] < x['M'] else 'Women')['sum'] + ' order more frequently'
#Result: 'Women order more frequently'.
sns.barplot(data=res_3, x='sex', y='sum', palette='tab10')

#-------------------------------------------------------------------------------
#QUESTION 4. How many categories of pizzas are there in the catalog?
#-------------------------------------------------------------------------------
res_4 = pizza_types['category'].unique()
print(res_4)

#-------------------------------------------------------------------------------
#QUESTION 5. How many pizzas belong to each cateogy?
#-------------------------------------------------------------------------------
#Easy way
pizza_types['category'].value_counts()
#Difficut way
pizza_types.groupby('category')['pizza_type_id'].agg(['count']).sort_values('count', ascending=False)

#As we can see, the result is the same. By using the easy way (value_counts function) we get the result immediately. 
#On the other hand, we use the groupby function to group by the pizza cateogry column, and then we use the count function
#in column 'pizza_type_id'. In the end, data is sorted by the count column from max to min. The difference between 
#these two methods is the result type. With value_counts, we get a Series object. With the second way, we get a DataFrame.

#-------------------------------------------------------------------------------
#QUESTION 6. Let's suppose a customer doen't want to eat pizzas with garlic. 
#Let's find them!
#-------------------------------------------------------------------------------
res_6 = pizza_types['ingredients'].apply(lambda x: 'Garlic' not in x)
print(pizza_types[res_6])
# Apply funtion is used to search every row in 'ingredients' column and returned True if 
#'Garlic' wasn't in column, else False.

#-------------------------------------------------------------------------------
#QUESTION 7. Show pizza name, size and cost for each pizza
#-------------------------------------------------------------------------------
res_7 = pizza_types.merge(pizzas, on='pizza_type_id')[['name', 'size', 'price']]
print(res_7)

#-------------------------------------------------------------------------------
#QUESTION 8. Show in a dataframe the products ordered where order_id=2 and
#then the total amount the customer must pay.
#-------------------------------------------------------------------------------
res_8 = all_info.query('order_id == 2')
print(res_8[['name', 'total_amount']])
res_8 = res_8.groupby('order_id')['total_amount'].agg(['sum'])
print(res_8['sum'].values[0])
#This query can be used where the shop owner wants to print the receipt.

#-------------------------------------------------------------------------------
#QUESTION 9. How many times each customer ordered pizza and what is the average 
#total price each one paid?
#-------------------------------------------------------------------------------
res_9 = all_info.groupby('cust_id')['total_amount'].agg(['count', 'mean'])
res_9 = res_9.merge(all_info[['cust_id', 'full_name']], on='cust_id')[['cust_id', 'full_name', 'count', 'mean']]
res_9 = res_9.drop_duplicates()
print(res_9)

#-------------------------------------------------------------------------------
#QUESTION 10. The shop owner wants to provide and extra discount to top 5 
#customers. These customers have paid the highest amounts.
#-------------------------------------------------------------------------------
res_10 = all_info.groupby('cust_id')['total_amount'].agg(['sum']).sort_values('sum', ascending=False)
res_10 = res_10.merge(all_info[['cust_id', 'full_name']], on='cust_id', how='left')
res_10 = res_10.drop_duplicates()
res_10 = res_10[['cust_id', 'full_name', 'sum']]
res_10.set_index('cust_id', drop=True, inplace=True)
print(res_10)

#-------------------------------------------------------------------------------
#QUESTION 11. How many orders delivered per month? Show the results in a 
#graph using seaborn
#-------------------------------------------------------------------------------
#Cast date column from object to date
orders['date'] = pd.to_datetime(orders['date'])
#Create a new column in order dataframe that contains only the month
orders['month'] = orders['date'].apply(lambda x: x.month)
#Count how many orders there are in the dataframe per month.
step = orders.groupby('month')['date'].agg(['count'])
#Give a short name to every month
months = pd.DataFrame(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], columns=['months'])
#Create a new column with number per month
months['month'] = np.arange(1,13)
months.set_index('month', drop=True, inplace=True)
#Merge dataframes and sort the values.
res_11 = months.merge(step, how='left', on='month')
res_11 = res_11.sort_values('count', ascending=False)
#Plot graph
plt.figure(figsize=(15,8))
sns.barplot(data=res_11, x='months', y = 'count', palette='tab10')
plt.xticks(ticks=list(np.arange(0, 12)), labels=list(res_11['months']))