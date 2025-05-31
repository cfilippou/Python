# The Pizza restaurant!

![1](https://github.com/user-attachments/assets/9e52aaae-9c04-44c6-8db4-d92cefc67c52)


The image presents the structure of a relational database composed of five CSV files: customers, orders, order_details, pizzas, and pizza_types. Each file includes specific columns with information and is linked to others through primary and foreign keys, forming a cohesive system for managing pizza orders.

The customers file contains customer details such as number, first name, and last name. Customers are connected to the orders file via the cid field. The orders file includes order records with fields like order_id, cid, date, and time. Each order can have multiple related entries in the order_details file, which includes order_detailsid, pizza id, order id, and quantity.

The pizzas file describes the available pizzas with attributes such as pizza_id, pizza_type_id, size, and price. Finally, the pizza_types file includes pizza type information with fields for ID, name, category, and ingredients. The relationships among the tables are mostly one-to-many (1:N), allowing for clear and consistent organization and analysis of data related to customers, orders, and products.
