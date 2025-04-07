import os
import pandas as pd
import mysql.connector

df=pd.read_csv("C:\\Users\\HP\\Desktop\\tips.csv")

import mysql.connector

# Database connection details
config = {
    'user': 'root',  # Replace with your MySQL username
    'password': 'ik565375#',  # Replace with your MySQL password
    'host': 'localhost',  # Replace with your MySQL host
    'database': 'db1',  # Replace with your database name
    'raise_on_warnings': True
}

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()


create_table_query = """
    CREATE TABLE IF NOT EXISTS tips (
    id INT AUTO_INCREMENT PRIMARY KEY,
    total_bill FLOAT,
    tip FLOAT,
    sex VARCHAR(10),
    smoker VARCHAR(10),
    day VARCHAR(10),
    time VARCHAR(10),
    size INT
);
    """

    # Execute the query
cursor.execute(create_table_query)
print("Table 'tips' created successfully!")



for index, row in df.iterrows():
    sql = """
    INSERT INTO tips (total_bill, tip, sex, smoker, day, time, size)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        row['total_bill'],
        row['tip'],
        row['sex'],
        row['smoker'],
        row['day'],
        row['time'],
        row['size']
        )
    cursor.execute(sql, values)
cnx.commit()