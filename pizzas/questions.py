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

#-------------------------------------------------------------------------------------------------------
# The following sections address specific questions based on the data contained in the CSV files. 
# A detailed description of the file structure can be found in the README.md.
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Q1. Find all customers with the last name "Taylor"
#-------------------------------------------------------------------------------------------------------
Taylor = customers[customers['Lastname'] == 'Taylor']

#-------------------------------------------------------------------------------------------------------
# Q2. How many pizzas are there in "Large" size?
#-------------------------------------------------------------------------------------------------------
large = pizzas[pizzas['size'] == 'L']

#-------------------------------------------------------------------------------------------------------
# Q3. How many pizzas are there in "Large" size?
#-------------------------------------------------------------------------------------------------------
# Create a list of pizza names that contain "Mushrooms" in their ingredients
# zip() pairs each pizza name with its corresponding ingredients list
mushrooms = [name for name, ingredients in zip(pizza_types['name'], pizza_types['ingredients']) if "Mushrooms" in ingredients]
for name in mushrooms:
  print(name)

#-------------------------------------------------------------------------------------------------------
# Q4. Find orders that were placed on 15/01/2015
#-------------------------------------------------------------------------------------------------------
orders['date'] = pd.to_datetime(orders['date'])
orders_15012025 = orders[orders['date'] == '2015-01-15']

#-------------------------------------------------------------------------------------------------------
# Q5. Find the average price of pizzas by size
#-------------------------------------------------------------------------------------------------------
pizzas.groupby('size')['price'].agg('mean')

#-------------------------------------------------------------------------------------------------------
# Q6. Find all pizzas that cost more than 15 euros
#-------------------------------------------------------------------------------------------------------
pizzas[pizzas['price'] > 15]

#-------------------------------------------------------------------------------------------------------
# Q7. Who are the 5 customers with the most orders?
#-------------------------------------------------------------------------------------------------------
step_1 = customers.merge(orders, how='left', left_on = 'id', right_on = 'cid')
step_2 = step_1.groupby(by = ['firstname', 'lastname'])['order_id'].agg('count')
step_2 = step_2.sort_values(ascending = False)
step_3 = step_2[:5]

#-------------------------------------------------------------------------------------------------------
# Q8. Which 5 pizza type has been ordered the most times?
#-------------------------------------------------------------------------------------------------------
step_1 = order_details.merge(pizzas, how='left', left_on='pizzaid', right_on='pizza_id')
step_2 = step_1.merge(pizza_types, how='left', left_on='pizza_type_id', right_on='pizza_type_id')
step_3 = step_2.groupby('name')['pizza_type_id'].agg('count').sort_values(ascending=False)
print(step_3[:5])

#-------------------------------------------------------------------------------------------------------
# Q14. For each customer, find their favorite pizza category (the one they have ordered the most)
#-------------------------------------------------------------------------------------------------------
# Step 1: Join order_details with orders table to get customer information
# Left join ensures we keep all order details and add corresponding customer IDs
step_1 = order_details.merge(orders, how='left', left_on = 'orderid', right_on = 'order_id')

# Step 2: Select only the relevant columns needed for the analysis
# Keep pizza ID, customer ID, and order ID for grouping and counting
step_2 = step_1[['pizzaid', 'cid', 'orderid']]

# Step 3: Group by customer and pizza ID, then count the number of orders
# This gives us how many times each customer ordered each specific pizza
step_3 = step_2.groupby(['cid', 'pizzaid'])['orderid'].agg('count')

# Step 4: Convert the grouped result back to a regular DataFrame
# Reset index to make cid and pizzaid regular columns again
step_4 = step_3.reset_index()

# Rename the count column to be more descriptive
step_4.rename(columns={'orderid': 'count'}, inplace=True)

# Sort by customer ID (ascending) and count (descending)
# This puts each customer's most ordered pizza at the top of their group
step_4 = step_4.sort_values(['cid', 'count'], ascending=[True, False])

# Step 5: Find the index of the maximum count for each customer
# idxmax() returns the index of the row with the highest count for each customer group
idx = step_4.groupby('cid')['count'].idxmax()

# Extract the rows with the highest pizza count for each customer
# This gives us each customer's favorite pizza (most frequently ordered)
step_5 = step_4.loc[idx, ['cid', 'pizzaid', 'count']]

#-------------------------------------------------------------------------------------------------------
# Q16. Which customers have ordered all available pizza types?
#-------------------------------------------------------------------------------------------------------
# Merge customers with orders using left join on customer id
step_1 = customers.merge(orders, how='left', left_on='id', right_on='cid')

# Merge the result with order_details using left join on order_id
step_2 = step_1.merge(order_details, how='left', left_on = 'order_id', right_on='orderid')

# Group by customer id and pizza id, then count occurrences and reset index
step_3 = step_2.groupby(['id', 'pizzaid'])['cid'].agg(['count']).reset_index()

# Group by customer id and count unique pizza types per customer
step_4 = step_3.groupby('id')['pizzaid'].agg(['count'])

# Sort customers by pizza type count in descending order
step_4 = step_4.sort_values(by=['count'], ascending = False)

# Filter customers who have ordered all available pizza types
res = step_4.query(f'count == {pizzas.shape[0]}')
#-------------------------------------------------------------------------------------------------------
# Q19. Find "dead hour" intervals where we have no orders, lasting at least 2 hours
#-------------------------------------------------------------------------------------------------------
# Convert the 'time' column to datetime format using the specified time format
# 'errors='coerce'' handles any invalid time values by converting them to NaT (Not a Time)
orders['time'] = pd.to_datetime(orders['time'], format='%H:%M:%S', errors='coerce')

# Create a categorical variable 'cat' that groups hours into 2-hour periods
# Using integer division (//), hour 0-1 becomes 0, hour 2-3 becomes 1, etc.
orders['cat'] = orders['time'].apply(lambda x: x.hour // 2)

# Count the number of orders in each 2-hour category and convert to DataFrame
# value_counts() returns the frequency of each category, reset_index() makes it a proper DataFrame
df = pd.DataFrame(orders['cat'].value_counts()).reset_index()

# Create a reference DataFrame with all possible 2-hour categories (0 to 11)
# This represents all 12 two-hour periods in a 24-hour day
cats = pd.DataFrame({'cat': [x for x in range(12)]})

# Find categories that have no orders by checking which categories are missing from the results
# Using boolean indexing with isin() to find categories not present in the order data
diff = cats[~cats['cat'].isin(df['cat'])]

# Create readable time range labels for the periods without orders
# Format: "HH:00 - HH:00" (e.g., "00:00 - 02:00", "02:00 - 04:00")
diff['range'] = diff['cat'].apply(lambda x: f'{x*2:02d}:00 - {x*2 + 2:02d}:00')

# Display the results showing which 2-hour periods had no orders
print('Two-hour periods without orders')
print(diff)



#-------------------------------------------------------------------------------------------------------
# Q20. Calculate the "customer lifetime value" for each customer, based on their purchase history
#-------------------------------------------------------------------------------------------------------
# Convert the 'date' column to datetime format using the specified date format (YYYY-MM-DD)
# 'errors='coerce'' handles any invalid date values by converting them to NaT (Not a Time)
orders['date'] = pd.to_datetime(orders['date'], format='%Y-%m-%d', errors='coerce')

# Step 1: Calculate the customer lifetime span (first and last order dates)
# Group by customer ID and find the minimum (first) and maximum (last) order dates
# This gives us the time period each customer has been active
step_1 = orders.groupby('cid')['date'].agg(['min', 'max'])

# Step 2: Calculate total monetary value per customer
# First, merge order details with pizza data to get pricing information
step_2 = order_details.merge(pizzas, how='left', left_on='pizzaid', right_on='pizza_id')

# Calculate the total price for each order line (quantity × unit price)
step_2['total_price'] = step_2['quantity'] * step_2['price']

# Merge with orders table to get customer information for each order detail
step_2 = step_2.merge(orders, how='left', left_on='orderid', right_on='order_id')

# Group by customer and sum up all their order values to get total spending per customer
step_2 = step_2.groupby('cid')['total_price'].agg(['sum'])

# Merge the monetary data with the date range data from step_1
step_2 = step_2.merge(step_1, how='right', left_on='cid', right_on='cid').reset_index()

# Step 3: Calculate order frequency per customer
# Count the total number of orders for each customer
step_3 = orders.groupby('cid')['order_id'].agg('count').reset_index()

# Combine frequency data with the monetary value and date range data
# This creates a comprehensive dataset with all CLV components: frequency, monetary value, and time span
step_3 = step_3.merge(step_2, how="left", on='cid')










