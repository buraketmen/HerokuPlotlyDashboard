import pandas as pd
import numpy as np
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn import preprocessing
import makeituseful as miu
#import requests

def data_manipulation():
    global df, numeric_features, features, old_df, df_for_feature, corr, annotations, pie_features, df_pie
    df.drop(['phone number'], axis=1, inplace=True)
    old_df = df.copy()
    for col in old_df.columns:
        old_df[col] = old_df[col].astype(str)

    features = df.columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    numeric_features = newdf.columns
    df_pie = df.copy()

    df['total minutes'] = df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes']
    df['total calls'] = df['total day calls'] + df['total eve calls'] + df['total night calls'] + df['total intl calls']
    df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']
    df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})
    df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})

    state_list = df['state'].unique().tolist()
    state_mapping = dict(zip(state_list, range(len(state_list))))
    df.replace({'state': state_mapping}, inplace=True)

    selected_columns = ['state', 'total calls', 'customer service calls', 'total minutes', 'total charge',
                        'account length', 'area code', 'international plan', 'voice mail plan', 'number vmail messages']
    new_df = df.filter(
        ['churn', 'state', 'customer service calls', 'total minutes', 'total calls', 'total charge', 'account length',
         'area code', 'international plan', 'voice mail plan', 'number vmail messages'], axis=1)

    newdata = new_df.values
    columns = new_df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    newdata_scaled = min_max_scaler.fit_transform(newdata)
    df_for_feature = pd.DataFrame(data=newdata_scaled, columns=columns)
    df_for_feature['churn'] = new_df['churn']
    corr = df_for_feature.corr()
    annotations = []
    z = corr.values.tolist()
    for n, row in enumerate(z):
        for m, val in enumerate(row):
            annotations.append(go.layout.Annotation(text=str(round(z[n][m], 2)), x=df_for_feature.columns[m],
                                                    y=df_for_feature.columns[n],
                                                    xref='x1', yref='y1', showarrow=False))

    df_pie['account length'] = df_pie['account length'].apply(miu.zerotofivehundred)
    df_pie['number vmail messages'] = df_pie['number vmail messages'].apply(miu.zerotoonehundred)
    df_pie['total day minutes'] = df_pie['total day minutes'].apply(miu.zerotofivehundred)
    df_pie['total eve minutes'] = df_pie['total eve minutes'].apply(miu.zerotofivehundred)
    df_pie['total night minutes'] = df_pie['total night minutes'].apply(miu.zerotofivehundred)
    df_pie['total intl minutes'] = df_pie['total intl minutes'].apply(miu.zerotoonehundred)
    df_pie['total day calls'] = df_pie['total day calls'].apply(miu.zerotofivehundred)
    df_pie['total eve calls'] = df_pie['total eve calls'].apply(miu.zerotofivehundred)
    df_pie['total night calls'] = df_pie['total night calls'].apply(miu.zerotofivehundred)
    df_pie['total intl calls'] = df_pie['total intl calls'].apply(miu.zerotoonehundred)
    df_pie['total day charge'] = df_pie['total day charge'].apply(miu.zerotoonehundred)
    df_pie['total eve charge'] = df_pie['total eve charge'].apply(miu.zerotoonehundred)
    df_pie['total night charge'] = df_pie['total night charge'].apply(miu.zerotoonehundred)
    df_pie['total intl charge'] = df_pie['total intl charge'].apply(miu.zerototen)


df = pd.read_csv('ChurnTelecoms.csv', sep=',')
numeric_features = None
features = None
old_df = None
df_for_feature = None
corr = None
annotations = None
pie_features = None
df_pie = None
data_manipulation()

graph_feature = {
    'background': '#f2f2f2',
    'text': '#1F2121'}


loadingtype = ['graph', 'cube', 'circle', 'dot', 'default']
loatype = 'graph'
external = ['https://codepen.io/buraketmen/pen/ExYgrKb.css']
app = dash.Dash(__name__,external_stylesheets=external)
#app = dash.Dash(__name__)
#USERNAME_PASSWORD_PAIRS = [['burak', '123'],['admin', 'admin']]
#auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H6("CHURN IN TELECOM'S DASHBOARD by BURAK KETMEN"),
        #html.Img(src="https://i.hizliresim.com/86aO61.png")
    ],id="banner",className="banner"),
    html.Div([
            html.Pre(
        id='counter_text',
        children=''
    ),
    dcc.Interval(
        id='interval-component',
        interval=900000,  # 900000 milliseconds = 15 min
        # interval=10000,
        n_intervals=0
    ),
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='xaxis',
                    options=[{'label': i.title(), 'value': i} for i in numeric_features],
                    value=numeric_features[0]
                ),
                dcc.RadioItems(
                    id='xaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ],id="dropdown-xaxis",className="twocolumnsblock"),

            html.Div([
                dcc.Dropdown(
                    id='yaxis',
                    options=[{'label': i.title(), 'value': i} for i in numeric_features],
                    value= numeric_features[0]
                ),
                dcc.RadioItems(
                    id='yaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ],id="dropdown-yaxis",className="twocolumnsblock"),

            dcc.Loading(id="loading-graphic",
                    children=[html.Div(
                        dcc.Graph(id='feature-graphic', config={
                            'toImageButtonOptions': {'width': 1500, 'height': 1300, 'format': 'png',
                                                     'filename': 'feature_graphic'}}))],
                    type=loatype)
        ],id="graphic",className="five columns"),
        html.Div([
            dcc.Loading(id="loading-heatmap",
                children=[html.Div([
                    dcc.Graph(id='feature-heatmap', config={'toImageButtonOptions':
                                                                {'width': 1500,
                                                                 'height': 1300,
                                                                 'format': 'png',
                                                                 'filename': 'heatmap'}})],id="heatmap-children")],
                        type=loatype)],id="heatmap",className="seven columns"),
    ],id="top-row"),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-histogram',
                options=[{'label': i.title(), 'value': i} for i in features],
                value=features[0]
            ),
            dcc.Loading(id="loading-histogram",
                        children=[html.Div(dcc.Graph(id='feature-histogram', config={'toImageButtonOptions':
                                                                                         {'width': 1500,
                                                                                          'height': 1300,
                                                                                          'format': 'png',
                                                                                          'filename': 'histogram'}}))],
                        type=loatype)
        ],id="histogram",className="seven columns"),

        html.Div([
            dcc.Dropdown(
                id='axis-box',
                options=[{'label': i.title(), 'value': i} for i in numeric_features],
                value=numeric_features[0]
            ),
            dcc.Loading(id="loading-box",
                        children=[html.Div(
                            dcc.Graph(id='feature-box', config={'toImageButtonOptions':
                                                                    {'width': 1500,
                                                                     'height': 1300,
                                                                     'format': 'png',
                                                                     'filename': 'box'}}))], type=loatype)
        ],id="box", className="five columns")
    ],id="medium-row"),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='axis-map',
                options=[{'label': i.title(), 'value': i} for i in numeric_features],
                value=numeric_features[0],
            ),
        ],id="dropdown-map",className="dropdown"),

        dcc.Loading(id="loading-map",
                    children=[html.Div([
                        dcc.Graph(id='feature-map', config={'toImageButtonOptions':
                                                                {'width': 1500,
                                                                 'height': 1300,
                                                                 'format': 'png',
                                                                 'filename': 'map'}})],id="map-children")], type=loatype)
    ],id="bottom-row"),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='axis-pie',
                options=[{'label': i.title(), 'value': i} for i in features],
                value=features[1:3],
                multi=True
            ),
        ],id="dropdown-pie",className="dropdown"),
        html.Div([
            dcc.Loading(id="loading-pie",
                        children=
                        html.Div(id='feature-pie'), className="row", type=loatype)
        ],id="loading-pie-children")
    ],id="pie")
    ], className="app_main_content"),

],className="container scalable")

@app.callback(Output("loading-graphic", "children"))
@app.callback(Output("loading-histogram", "children"))
@app.callback(Output("loading-box", "children"))
@app.callback(Output("loading-heatmap", "children"))
@app.callback(Output("loading-map", "children"))
@app.callback(Output("loading-pie", "children"))


@app.callback(Output('counter_text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_data(n):
    global df
    df = pd.read_csv('ChurnTelecoms.csv', sep=',')
    data_manipulation()

@app.callback(
    Output('feature-pie', 'children'),
    [Input('axis-pie', 'value'), Input('interval-component', 'n_intervals')],
    [State('feature-pie', 'children')])
def update_pie(xaxis_name,n,s):
    xaxis_name = list(xaxis_name)
    graphs= []
    if len(xaxis_name) > 2:
        class_choice = 'four columns'
    elif len(xaxis_name) == 2:
        class_choice = 'six columns'
    else:
        class_choice = 'twelve columns'

    for axis in xaxis_name:
        data = go.Pie(labels=df_pie[axis].unique().tolist(),values=df_pie.groupby([axis]).size().tolist())
        graphs.append(html.Div(dcc.Graph(
            id=axis,
            animate=True,
            figure={'data': [data],
                    'layout' : go.Layout(margin={'l':50,'r':50,'t':50,'b':50},
                                                        title='{}'.format(axis),
                                         plot_bgcolor=graph_feature['background'],
                                         paper_bgcolor=graph_feature['background']
                                         )},), className=class_choice))
    return graphs

@app.callback(
Output('feature-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('feature-map', 'figure')]
)
def update_heatmap(n,s):
    return {
        "data": [go.Heatmap(x=df_for_feature.columns, y=df_for_feature.columns, z=corr.values.tolist(),
                            colorscale='Electric', showscale=True)],
        "layout": go.Layout(margin={'l': 200, 'b': 200, 't': 50, 'r': 100},
                            annotations=annotations,
                            plot_bgcolor=graph_feature['background'],
                            paper_bgcolor=graph_feature['background']
                            )
    }

@app.callback(
    Output('feature-map', 'figure'),
    [Input('axis-map', 'value'), Input('interval-component', 'n_intervals')],
    [State('feature-map', 'figure')]
    )
def update_map(axis_name,n,s):
    country = {"AL": "Alabama","AK": "Alaska","AS": "American Samoa","AZ": "Arizona","AR": "Arkansas","CA": "California",
    "CO": "Colorado","CT": "Connecticut","DE": "Delaware","DC": "District Of Columbia","FM": "Federated States Of Micronesia","FL": "Florida",
    "GA": "Georgia","GU": "Guam","HI": "Hawaii","ID": "Idaho","IL": "Illinois","IN": "Indiana","IA": "Iowa","KS": "Kansas",
    "KY": "Kentucky","LA": "Louisiana","ME": "Maine","MH": "Marshall Islands","MD": "Maryland","MA": "Massachusetts","MI": "Michigan",
    "MN": "Minnesota","MS": "Mississippi","MO": "Missouri","MT": "Montana","NE": "Nebraska","NV": "Nevada","NH": "New Hampshire","NJ": "New Jersey","NM": "New Mexico",
    "NY": "New York","NC": "North Carolina","ND": "North Dakota","MP": "Northern Mariana Islands","OH": "Ohio","OK": "Oklahoma","OR": "Oregon",
    "PW": "Palau","PA": "Pennsylvania","PR": "Puerto Rico","RI": "Rhode Island","SC": "South Carolina","SD": "South Dakota",
    "TN": "Tennessee","TX": "Texas","UT": "Utah","VT": "Vermont","VI": "Virgin Islands","VA": "Virginia","WA": "Washington",
    "WV": "West Virginia","WI": "Wisconsin","WY": "Wyoming"}

    dff = df.groupby(['state']).mean().reset_index()
    dff_max = df.groupby(['state']).max().reset_index()
    dff_max['state'] =old_df['state'].unique()

    dff_min = df.groupby(['state']).min().reset_index()
    dff_min['state'] = old_df['state'].unique()

    dff['state'] =old_df['state'].unique()
    dff['fullname'] = dff['state'].map(country)
    dff['text'] = dff['fullname'] + '<br>' + \
                 'Max: ' + dff_max[axis_name].astype(str) + '<br>' + \
                 'Min: ' + dff_min[axis_name].astype(str)

    return {
        'data': [go.Choropleth(locations= dff['state'], z = dff[axis_name], locationmode='USA-states', text= dff['text'],
                               colorbar={'title': {"text": "Average", "side": "top"}})],
        'layout': go.Layout(title=axis_name.title(),
                            margin={'l': 50, 'b': 25, 't': 25, 'r': 100},
                            geo={'scope':'usa'},
                            plot_bgcolor=graph_feature['background'],
                            paper_bgcolor=graph_feature['background'],
                            font={'color': graph_feature['text']}
                            )
    }

@app.callback(
    Output('feature-histogram', 'figure'),
    [Input('xaxis-histogram', 'value'), Input('interval-component', 'n_intervals')],
    [State('feature-histogram', 'figure')])
def update_histogram(xaxis_name,n,s):
    traces = []
    for churn_name in old_df['churn'].unique().tolist():
        df_by_churn = old_df[old_df['churn'] == churn_name]
        traces.append(go.Histogram(
            x=df_by_churn[xaxis_name],
            opacity=1,
            name=churn_name
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 100, 'b': 100, 't': 100, 'r': 100},
            #hovermode='closest',
            # barmode = 'overlay',
            xaxis={'title': xaxis_name.title()},
            yaxis={'title': 'Count'},
            plot_bgcolor= graph_feature['background'],
            paper_bgcolor=graph_feature['background'],
            font={'color':graph_feature['text']}
        )
    }

@app.callback(
    Output('feature-box', 'figure'),
    [Input('axis-box', 'value'), Input('interval-component', 'n_intervals')],
    [State('feature-box', 'figure')])
def update_histogram(xaxis_name,n,s):
    traces = []
    for churn_name in df['churn'].unique().tolist():
        df_by_churn = df[df['churn'] == churn_name]
        traces.append(go.Box(
            y=df_by_churn[xaxis_name],
            opacity=1,
            name=churn_name
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 100, 'b': 100, 't': 100, 'r': 100},
            plot_bgcolor=graph_feature['background'],
            paper_bgcolor=graph_feature['background'],
            font={'color': graph_feature['text']}
        )
    }

@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
     Input('yaxis', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value'), Input('interval-component', 'n_intervals')],
    [State('feature-graphic', 'figure')])
def update_graph(xaxis_name, yaxis_name, xaxis_type, yaxis_type,n,s):
    traces = []
    for churn_name in df['churn'].unique().tolist():
        df_by_churn =  df[df['churn'] == churn_name]
        traces.append(go.Scatter(
            x=df_by_churn[xaxis_name],
            y=df_by_churn[yaxis_name],
            text=df['churn'],
            mode='markers',
            opacity=1,
            marker={'size': 10},
            name=churn_name
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': xaxis_name.title(),
                   'type': 'linear' if xaxis_type == 'Linear' else 'log'},
            yaxis={'title': yaxis_name.title(),
                   'type': 'linear' if yaxis_type == 'Linear' else 'log'},
            margin={'l': 100, 'b': 100, 't': 100, 'r': 100},
            hovermode='closest',
            plot_bgcolor=graph_feature['background'],
            paper_bgcolor=graph_feature['background'],
            font={'color': graph_feature['text']}
        )
    }

if __name__ == '__main__':
    app.run_server()