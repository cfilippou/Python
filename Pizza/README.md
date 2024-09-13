# Pizza analysis
## Title
A Python project demonstrating data analysis techniques using pandas for decision-making insights.

## Description
Project Summary: Data Analysis in Python Using Pandas

This project demonstrates the application of data analysis techniques in Python, utilizing the powerful pandas library. The goal is to provide a comprehensive guide on how to analyze datasets and extract meaningful insights, leading to informed decision-making. Throughout the project, various examples are presented to illustrate key functions and methods that streamline the data analysis process.

The following core functionalities of pandas were extensively used:
1. Data Merging (merge): Combining multiple datasets to create a unified structure for analysis.
2. Grouping and Aggregation (groupby, agg): Summarizing data by grouping based on specific columns and applying aggregation functions (e.g., sum, mean, count) to analyze patterns across groups.
3. Removing Duplicates (drop_duplicates): Cleaning the data by eliminating redundant rows to ensure accuracy in analysis.
4. Renaming Columns (rename): Enhancing data readability by renaming columns with more meaningful labels.
5. Custom Functions with apply and Lambda Expressions: Creating flexible, custom transformations on columns using lambda functions and the apply method.
6. Counting Values (value_counts): Understanding data distributions by counting unique values in specific columns.
7. Querying Data (query): Filtering datasets based on complex conditions to focus on relevant subsets of data.

Through these techniques, the project showcases how to manipulate and explore real-world datasets, offering insights such as customer segmentation, sales trends, and performance metrics. By combining multiple operations, the project illustrates how to transform raw data into actionable knowledge, allowing businesses or individuals to make data-driven decisions with confidence.

## Parts
1. Import libraries
2. DataFrame Overview
    1. pizza
    2. pizza_types
    3. orders
    4. order_details.csv
    5. customers.csv
3. Merge all dataframes into one
4. Questions
   1. Which pizza size is most preferred by men and women? Show the result in a graph with seaborn.
   2. Create a new dataframe that contains id, full name, sex, total orders, total paid amound.
   3. Do men or women order more? Show the results in a barplot.
   4. How many categories of pizzas are there in the catalog?
   5. How many pizzas fall into each category?
   6. Let's suppose a customer doen't want to eat pizzas with garlic. Let's find them!
   7. Show pizza name, size and cost for each pizza.
   8. Show in a dataframe the products ordered where order_id=2 and then the total amount the customer must pay.
   9. How many times each customer ordered pizza and what is the average total price each one paid?
   10. The shop owner wants to provide and extra discount to top 5 customers. These customers have paid the highest amounts.
   11. How many orders delivered per month? Show the results in a graph using seaborn.
  
   ## Conclusion
   The primary goal of this project was to help new programmers understand and apply fundamental data analysis functions using
   Python's pandas library. By providing clear examples and practical use cases, I aimed to demonstrate how common functions like merge,
   groupby, apply, and others can be used to efficiently manipulate and analyze data. This project serves as an introduction to data analysis,
   empowering beginners to explore real-world datasets, gain insights, and develop a strong foundation for more advanced techniques in the future.
