import dash
import dash_auth
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from stockstats import StockDataFrame as Sdf
import dash_bootstrap_components as dbc
from dash import dash_table as dt
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random
import time
import pandas as pd
import numpy as np
import json
import threading
from price_generator import PriceGenerator

global RUN_FLAG

pg = PriceGenerator()

dfs = pg.load_data()
TICKERS = list(dfs)
RUN_FLAG = True


# defining style color
colors = {"background": "#000000", "text": "#ffFFFF"}
external_stylesheets = [dbc.themes.SLATE]

with open(r'ins\users.json', 'r') as f:
    VALID_USERNAME_PASSWORD_PAIRS = json.load(f)
f.close()

# adding css
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# AUTH
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
app.config['suppress_callback_exceptions'] = True


app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(
            [  # header Div
                dbc.Row(
                    [
                        dbc.Col(
                            html.Header(
                                [
                                    html.H1(
                                        "Ticker Generator Dashboard",
                                        style={
                                            "textAlign": "center",
                                            "color": colors["text"],
                                        },
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Div(
            [  # Dropdown Div
                dbc.Row(
                    [
                        dbc.Col(  # Tickers
                            dcc.Dropdown(
                                id="stock_name",
                                options=[
                                    {
                                        "label": str(TICKERS[i]),
                                        "value": str(TICKERS[i]),
                                    }
                                    for i in range(len(TICKERS))
                                ],
                                searchable=True,
                                value=str(
                                    random.choice(
                                        TICKERS
                                    )
                                ),
                                placeholder="enter stock name",
                            ),
                            width={"size": 3, "offset": 3},
                        ),
                        dbc.Col(  # Graph type
                            dcc.Dropdown(
                                id="chart",
                                options=[
                                    {"label": "price", "value": "Price"},
                                    {"label": "close", "value": "Close"},
                                    {"label": "candlestick",
                                     "value": "Candlestick"},
                                    {"label": "Simple moving average",
                                     "value": "SMA"},
                                    {
                                        "label": "Exponential moving average",
                                        "value": "EMA",
                                    },
                                    {"label": "MACD", "value": "MACD"},
                                    {"label": "RSI", "value": "RSI"},
                                    {"label": "OHLC", "value": "OHLC"},
                                ],
                                value="Price",
                                style={"color": "#000000"},
                            ),
                            width={"size": 3},
                        ),
                        dbc.Col(  # button
                            dbc.Button(
                                "Plot",
                                id="submit-button-state",
                                className="mr-1",
                                n_clicks=1,
                            ),
                            width={"size": 2},
                        ),
                    ]
                )
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="live price",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                            )
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="graph",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                            ),
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dt.DataTable(
                                id="info",
                                style_table={"height": "auto"},
                                style_cell={
                                    "white_space": "normal",
                                    "height": "auto",
                                    "backgroundColor": colors["background"],
                                    "color": "white",
                                    "font_size": "16px",
                                },
                                style_data={"border": "#4d4d4d"},
                                style_header={
                                    "backgroundColor": colors["background"],
                                    "fontWeight": "bold",
                                    "border": "#4d4d4d",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "center"}
                                    for c in ["attribute", "value"]
                                ],
                            ),
                            width={"size": 6, "offset": 3},
                        )
                    ]
                ),
            ]
        ),
    ],
)


# Callback main graph on 1 sec interval or "Plot" button
@app.callback(
    # output
    [Output("graph", "figure"), Output("live price", "figure")],
    # input
    [Input("submit-button-state", "n_clicks"),
     Input('interval-component', 'n_intervals')],
    # state
    [State("stock_name", "value"), State("chart", "value")],
)
def graph_genrator(n_clicks, n_int, ticker, chart_name):

    global RUN_FLAG
    global dfs

    if not RUN_FLAG:
        raise PreventUpdate

    RUN_FLAG = False

    # loading data
    dfs = pg.load_data()
    df = dfs[ticker]

    stock = Sdf(df)

    # selecting graph type

    # Price plot
    if chart_name == "Price":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(df.index), y=list(df.price), fill="tozeroy", name="close"
                )
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )
    if chart_name == "Close":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(df.index), y=list(df.close), fill="tozeroy", name="close"
                )
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    # Candelstick
    if chart_name == "Candlestick":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=list(df.index),
                    open=list(df.open),
                    high=list(df.high),
                    low=list(df.low),
                    close=list(df.close),
                    name="Candlestick",
                )
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    # simple moving average
    if chart_name == "SMA":
        close_ma_10 = df.close.rolling(10).mean()
        close_ma_15 = df.close.rolling(15).mean()
        close_ma_30 = df.close.rolling(30).mean()
        close_ma_100 = df.close.rolling(100).mean()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(close_ma_10.index), y=list(close_ma_10), name="10 Days"
                ),
                go.Scatter(
                    x=list(close_ma_15.index), y=list(close_ma_15), name="15 Days"
                ),
                go.Scatter(
                    x=list(close_ma_30.index), y=list(close_ma_15), name="30 Days"
                ),
                go.Scatter(
                    x=list(close_ma_100.index), y=list(close_ma_15), name="100 Days"
                ),
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    # Open_high_low_close
    if chart_name == "OHLC":
        fig = go.Figure(
            data=[
                go.Ohlc(
                    x=df.index,
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                )
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    # Exponential moving average
    if chart_name == "EMA":
        close_ema_10 = df.close.ewm(span=10).mean()
        close_ema_15 = df.close.ewm(span=15).mean()
        close_ema_30 = df.close.ewm(span=30).mean()
        close_ema_100 = df.close.ewm(span=100).mean()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(close_ema_10.index), y=list(close_ema_10), name="10 Days"
                ),
                go.Scatter(
                    x=list(close_ema_15.index), y=list(close_ema_15), name="15 Days"
                ),
                go.Scatter(
                    x=list(close_ema_30.index), y=list(close_ema_30), name="30 Days"
                ),
                go.Scatter(
                    x=list(close_ema_100.index),
                    y=list(close_ema_100),
                    name="100 Days",
                ),
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    # Moving average convergence divergence
    if chart_name == "MACD":
        df["MACD"], df["signal"], df["hist"] = (
            stock["macd"],
            stock["macds"],
            stock["macdh"],
        )
        fig = go.Figure(
            data=[
                go.Scatter(x=list(df.index), y=list(df.MACD), name="MACD"),
                go.Scatter(x=list(df.index), y=list(
                    df.signal), name="Signal"),
                go.Scatter(
                    x=list(df.index),
                    y=list(df["hist"]),
                    line=dict(color="royalblue", width=2, dash="dot"),
                    name="Hitogram",
                ),
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

        # Relative strength index
    if chart_name == "RSI":
        rsi_6 = stock["rsi_6"]
        rsi_12 = stock["rsi_12"]
        fig = go.Figure(
            data=[
                go.Scatter(x=list(df.index), y=list(
                    rsi_6), name="RSI 6 Day"),
                go.Scatter(x=list(df.index), y=list(
                    rsi_12), name="RSI 12 Day"),
            ],
            layout={
                "height": 1000,
                "title": chart_name,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]},
                "uirevision": "1111"
            },
        )
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                activecolor="blue",
                bgcolor=colors["background"],
                buttons=list(
                    [
                        dict(count=10, label="10S", step="second", stepmode="backward"),
                        dict(count=60, label="1min", step="second", stepmode="backward"),
                        dict(count=300, label="5min", step="second", stepmode="backward"),
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=60, label="1H", step="minute", stepmode="backward"),
                        dict(count=24, label="1D", step="hour", stepmode="backward"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(count=5, label="5y", step="year",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(step="all"),
                    ]
                ),
            ),
        )

    res_df = dfs[ticker]

    if len(res_df) >= 2:
        price = res_df.price.iloc[-1]
        prev_close = res_df.price.iloc[-2]
    else:
        price = 0
        prev_close = 0

    live_price = go.Figure(
        data=[
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=price,
                mode="number+delta",
                title={"text": "Price"},
                delta={"reference": prev_close},
            )
        ],
        layout={
            "height": 300,
            "showlegend": True,
            "plot_bgcolor": colors["background"],
            "paper_bgcolor": colors["background"],
            "font": {"color": colors["text"]},
            "uirevision": "1111"
        },
    )

    RUN_FLAG = True
    return fig, live_price


if __name__ == "__main__":

    t1 = threading.Thread(target=app.run_server,
                          kwargs={"debug": False, "use_reloader": False, "port": 66, "host": "localhost"})
    t2 = threading.Thread(target=pg.run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
