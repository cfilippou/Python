#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

#Load data
customers = pd.read_csv('customers.csv')
order_details = pd.read_csv('order_details.csv')
orders = pd.read_csv('orders.csv')
pizza_types = pd.read_csv('pizza_types.csv')
pizzas = pd.read_csv('pizzas.csv')

#-------------------------------------------------------------------------
# The following sections address specific questions based on the data contained in the CSV files. 
# A detailed description of the file structure can be found in the README.md.
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Q1. Find all customers with the last name "Taylor"
#-------------------------------------------------------------------------
Taylor = customers[customers['Lastname'] == 'Taylor']

#-------------------------------------------------------------------------
# Q2. How many pizzas are there in "Large" size?
#-------------------------------------------------------------------------
large = pizzas[pizzas['size'] == 'L']

#-------------------------------------------------------------------------
# Q3. How many pizzas are there in "Large" size?
#-------------------------------------------------------------------------
# Create a list of pizza names that contain "Mushrooms" in their ingredients
# zip() pairs each pizza name with its corresponding ingredients list
mushrooms = [name for name, ingredients in zip(pizza_types['name'], pizza_types['ingredients']) if "Mushrooms" in ingredients]
for name in mushrooms:
  print(name)

#-------------------------------------------------------------------------
# Q4. Find orders that were placed on 15/01/2015
#-------------------------------------------------------------------------
orders['date'] = pd.to_datetime(orders['date'])
orders_15012025 = orders[orders['date'] == '2015-01-15']

#-------------------------------------------------------------------------
# Q5. Find the average price of pizzas by size
#-------------------------------------------------------------------------
pizzas.groupby('size')['price'].agg('mean')

#-------------------------------------------------------------------------
# Q6. Find all pizzas that cost more than 15 euros
#-------------------------------------------------------------------------
pizzas[pizzas['price'] > 15]












#-------------------------------------------------------------------------
# Q19. Find "dead hour" intervals where we have no orders, lasting at least 2 hours
#-------------------------------------------------------------------------
orders['time'] = pd.to_datetime(orders['time'], format='%H:%M:%S', errors='coerce')
orders['cat'] = orders['time'].apply(lambda x: x.hour // 2)
df = pd.DataFrame(orders['cat'].value_counts()).reset_index()
cats = pd.DataFrame({'cat': [x for x in range(12)]})
diff = cats[~cats['cat'].isin(df['cat'])]
diff['range'] = diff['cat'].apply(lambda x: f'{x*2:02d}:00 - {x*2 + 2:02d}:00')
print('Two-hour periods without orders')
print(diff)















