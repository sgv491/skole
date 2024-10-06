import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()[1]
        df = pd.DataFrame(data)
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['year', 'value']].dropna()
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

def fetch_sp500_data(start_date="2010-01-01", end_date="2020-12-31"):
    ticker_symbol = "^GSPC"
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    sp500_data['Year'] = sp500_data.index.year
    annual_close = sp500_data['Close'].resample('Y').last()
    annual_growth_rate = annual_close.pct_change() * 100  # Convert to percentage
    df = pd.DataFrame({'Year': annual_close.index.year, 'Growth Rate': annual_growth_rate.values})
    return df.dropna()

def plot_static(df, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Growth Rate'], marker='o', linestyle='-', label=df['Indicator'].iloc[0])
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Growth Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.show()

def create_interactive_plot(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Growth Rate'], mode='lines+markers', name=title))
    fig.update_layout(title=title, xaxis_title='Year', yaxis_title='Growth Rate (%)', legend_title='Indicator')
    fig.show()