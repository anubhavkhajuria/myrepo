import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import psycopg2

# Connecting to database
conn = psycopg2.connect(database="your_database", user="your_user", password="your_password", host="your_host", port="your_port")
cursor = conn.cursor()

# Extract data from the database*
query = "SELECT * FROM your_table;"
cursor.execute(query)
data = cursor.fetchall()

# Close the database connection
cursor.close()
conn.close()

# Convert data to a pandas DataFrame
columns = ['column1', 'column2', 'column3', 'column4']
df = pd.DataFrame(data, columns=columns)

# Perform data cleaning and transformation
df = df.drop_duplicates()

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['column1', 'column2', 'column3', 'column4']])

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Perform clustering using K-means algorithm
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(reduced_data)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Save processed data to a new table in the database
conn = psycopg2.connect(database="your_database", user="your_user", password="your_password", host="your_host", port="your_port")
cursor = conn.cursor()

# Create a new table for processed data
create_table_query = '''
    CREATE TABLE processed_data (
        column1 datatype,
        column2 datatype,
        column3 datatype,
        column4 datatype,
        cluster_label integer
    );
'''
cursor.execute(create_table_query)

# Insert processed data into the table
for row in df.itertuples(index=False):
    insert_query = "INSERT INTO processed_data (column1, column2, column3, column4, cluster_label) VALUES (%s, %s, %s, %s, %s);"
    cursor.execute(insert_query, row)

conn.commit()
cursor.close()
conn.close()


