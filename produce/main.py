import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger
import numpy as np
from dotenv import load_dotenv



# Suppress joblib warning about physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to the number of cores you want to use

def plot_stocks(df, title="Stock Prices Over Time", figsize=(12, 6)):
    """
    Plot the stock prices from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock prices with dates as index
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def pca_plot(n_clusters, data, symbols):
    # run kmeans on reduced data
    try:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        labels = kmeans.predict(data)
    except Exception as e:
        logger.error(f"Error fitting kmeans: {e}")
        return
    # create DataFrame aligning labels & companies
    df = pd.DataFrame({'labels': labels, 'companies': symbols})

    # Define step size of mesh
    h = 0.005

    # plot the decision boundary
    x_min, x_max = data[:, 0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain abels for each point in the mesh using our trained model
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # define colorplot
    cmap = plt.cm.Paired

    # plot figure
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(Z, interpolation='nearest',
            extent = (xx.min(), xx.max(), yy.min(), yy.max()),
            cmap = cmap,
            aspect = 'auto', origin='lower')
    plt.plot(data[:, 0], data[:, 1], 'k.', markersize=5)

    # Add labels with offset
    for i in range(len(data)):
        plt.annotate(str(i), 
                    (data[i, 0], data[i, 1]),
                    xytext=(5, 5),  # offset from the point
                    textcoords='offset points')


    # plot the centroid of each cluster as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
    marker='x', s=50, linewidth=2,
    color='w', zorder=10)

    plt.title('K-Means Clustering on Stock Market Movements (PCA-Reduced Data)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()



def get_data(symbols):
    df = pd.DataFrame()
    for symbol in symbols:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}'
        r = requests.get(url)
        data = r.json()
        time_series = data['Time Series (Daily)']
        df_symbol = pd.DataFrame.from_dict(time_series, orient='index')
        df_close = df_symbol['5. adjusted close'].rename(symbol)  # Rename column to stock symbol
        df = pd.concat([df, df_close], axis=1)
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    # Sort by date
    df = df.sort_index()
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    df = df.rename(columns={
        'AAPL': 'Apple',
        'IBM': 'IBM',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'NVDA': 'Nvidia',
        'WMT': 'Walmart',
        'JNJ': 'Johnson & Johnson',
        'VZ': 'Verizon',
        'TM': 'AT&T',
        'PFE': 'Pfizer',
    })
    df.to_csv('all_close.csv')
    
    

def find_latest_time_stamp(data, year_month):
    """Find the latest timestamp for each day in January 2009"""
    return [max([ts for ts in data if f'{year_month}-{day:02d}' in ts]) 
            for day in range(1, 32) 
            if any(f'{year_month}-{day:02d}' in ts for ts in data)]

def intraday_data(symbols):
    logger.info("Getting intraday data for each symbol")
    """Get intraday data for each symbol"""
    # Get all the years and months from 2014 to 2024
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    years_month_builder = [f'{year}-{month}' for year in years for month in months]
    df = pd.DataFrame()
    data_list = {}
    for symbol in symbols:
        logger.info(f"Getting intraday data for {symbol}")
        for year_month in years_month_builder:

            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&month={year_month}&outputsize=full&apikey={API_KEY}'
            r = requests.get(url)
            data = r.json()
            

            # print(data['Time Series (60min)'].keys())
            date_time_list = find_latest_time_stamp(data['Time Series (60min)'].keys(), year_month)

            for date_time in date_time_list:
                data_list[date_time.split(' ')[0]] = data['Time Series (60min)'][date_time]['4. close']


    
    # print(data_list)
        df_symbol = pd.DataFrame.from_dict(data_list, orient='index', columns=[symbol])
     

        df = pd.concat([df, df_symbol], axis=1)

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    # Sort by date
    df = df.sort_index()
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    df = df.rename(columns={
        'AAPL': 'Apple',
        'IBM': 'IBM',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'NVDA': 'Nvidia',
        'WMT': 'Walmart',
        'JNJ': 'Johnson & Johnson',
        'VZ': 'Verizon',
        'TM': 'AT&T',
        'PFE': 'Pfizer',
    })
    df.to_csv('all_intraday.csv')
   
       
    # print(data['Time Series (60min)']['2009-01-08 11:00:00'])
    
def cluster_stocks(df):
    """Cluster stocks based on their price movements"""
    # Calculate daily returns
    returns = df.pct_change()
    
    # Drop the first row (NaN) and transpose to get stocks as rows
    returns = returns.dropna().T
    
    # Scale the data
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(scaled_returns)
    
    # Create a DataFrame with results
    results = pd.DataFrame({
        'Company': df.columns,
        'Cluster': labels
    })
    
    # Sort by cluster
    results = results.sort_values('Cluster')
    
    # Print results
    print("\nStock Clustering Results:")
    print("------------------------")
    for cluster in range(5):
        print(f"\nCluster {cluster}:")
        cluster_stocks = results[results['Cluster'] == cluster]['Company'].tolist()
        print(", ".join(cluster_stocks))
    
    return results

if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv('API_KEY')
    print("Hello, World!")

    AAPL = 'AAPL'
    IBM = 'IBM'
    MSFT = 'MSFT'
    GOOGL = 'GOOGL'
    AMZN = 'AMZN'
    TSLA = 'TSLA'
    NVDA = 'NVDA'
    WMT = 'WMT'
    JNJ = 'JNJ'
    VZ = 'VZ'
    TM = 'TM'
    PFE = 'PFE'

    symbols = [AAPL, IBM, MSFT, GOOGL, AMZN, TSLA, NVDA, WMT, JNJ, VZ, TM, PFE]
    # get_data(symbols)
    # intraday_data(symbols)

    # # Plot the data
    # df = pd.read_csv('all_intraday.csv').drop(columns=['Unnamed: 0'])
    # plot_stocks(df, title="Stock Prices Comparison")

  
    # str = ['2009-01-09 13:55:00', '2009-01-09 13:50:00']
    # new_str = [s for s in str if '2009-01-09' in s]
    # print(new_str)

    # dic = {'2024-04-01 19:00:00': '168.7032', '2024-04-02 19:00:00': '168.5539'}
    # df = pd.DataFrame.from_dict(dic, orient='index')
    # print(df)

    # # Get intraday data
    # df = intraday_data(symbols)
    
    # # Perform clustering
    # cluster_results = cluster_stocks(df)
