import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import io
import base64
import datetime
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression



print(dcc.__version__) # 0.6.0 or above is required

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
], style= {'textAlign': 'center', 'fontSize': 20})

###INDEX PAGE
index_page = html.Div([
    html.H1('Data Mining Project'),
    html.H3('Aira Domingo'),
    dcc.Link('Load Dataset', href='/page-1'),
    html.Br(),
    dcc.Link('Explore Data', href='/page-2'),
    html.Br(),
    dcc.Link('Models', href='/page-3')

])



###PAGE 1
page_1_layout = html.Div([
    html.H2('Load Dataset'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Br(),
    dcc.Link('Explore Data', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
], )

#csv into table
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('rows'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={
                'height': '300px',
                'overflowY': 'scroll',
                'border': 'thin lightgrey solid'
            },
        ),

        html.Hr(),  # horizontal line

        # Dataset General Information
        html.Div('General Dataset Information'),
        dcc.Dropdown(
            id= 'gen-info-dropdown',
            options =[
                {'label': 'Number of Observations', 'value': 'observations'},
                {'label': 'Number of Features', 'value': 'features'}
            ],
            value= 'observations'
        ),
        html.Div(id='gen-info-container')

    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



@app.callback(Output('gen-info-container', 'children'),
              [Input('gen-info-dropdown', 'value')])
def update_info(value):
    fp = pd.read_csv('NFA_2018.csv')
    if value == 'observations':
        return fp.size
    elif value == 'features':
        return len(fp.columns)

###PAGE 2
page_2_layout = html.Div([
    html.H3('Explore Data'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Raw Data', value='tab-1-example'),
        dcc.Tab(label='Preprocessed Data', value='tab-2-example'),
    ]),
    html.Div(id='tabs-content-example'),
    html.Br(),
    dcc.Link('Models', href='/page-3'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])


@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div([
            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'Total Ecological Footprint Consumption', 'value': 'total'},
                    {'label': 'Ecological Footprint of Carbon', 'value': 'carbon'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='total'
            ),

            dcc.Graph(
                id='year-vs-total-consumption')


            ])

    elif tab == 'tab-2-example':
        return html.Div([
            dcc.Dropdown(
                id='pre-dropdown',
                options=[
                    {'label': 'Total Ecological Footprint Consumption', 'value': 'total'},
                    {'label': 'Ecological Footprint of Carbon', 'value': 'carbon'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='total'
            ),
            dcc.Graph(
                id='preprocessed-data-graph'
            ),
            dcc.Dropdown(
                id='preprocessed-info-dropdown',
                options=[
                    {'label': 'Number of Observations', 'value': 'observations'},
                    {'label': 'Number of Features', 'value': 'features'}
                ],
                value='observations'
            ),
            html.Div(id='preprocessed-info-container')
        ])



#callback for scatter plot raw data
@app.callback(dash.dependencies.Output('year-vs-total-consumption','figure'),
              [dash.dependencies.Input('my-dropdown', 'value')])
def update_graph(value):
    data = pd.read_csv('NFA_2018.csv')
    df = data[data.record == 'EFConsTotGHA']
    return {
        'data': [
            go.Scatter(
                x=df[df['UN_region'] == i]['year'],
                y=df[df['UN_region'] == i][value],
                text=df[df['UN_region'] == i]['country'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=i
            ) for i in df.UN_region.unique()
        ],
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'Year'},
            yaxis={'title': 'Ecological Footprint Consumption' if value == 'total' else 'Carbon Emission'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
            title='Ecological Footprint Total Consumption from 1960-2014' if value == 'total' else 'Carbon Emission from 1960-2014',
            hovermode='closest'
        )
    }



#callback for scatter plot preprocessed data
@app.callback(dash.dependencies.Output('preprocessed-data-graph','figure'),
              [dash.dependencies.Input('pre-dropdown', 'value')])
def update_graph(value):
    data = pd.read_csv('NFA_2018.csv')
    footprintp = data[data.record == 'EFConsTotGHA']
    footprintp = footprintp[
        ['country', 'ISO alpha-3 code', 'UN_region', 'UN_subregion', 'year', 'record', 'crop_land', 'grazing_land',
         'forest_land', 'fishing_ground', 'built_up_land', 'population', 'carbon', 'total', 'Percapita GDP (2010 USD)']]

    # nan missing data
    footprintp = footprintp.replace(0, np.NaN)
    # drop rows with no data and data from world
    footprintp = footprintp[footprintp.crop_land.notna()]
    footprintp = footprintp[footprintp.country != 'World']

    # fill nan with 0
    footprintp = footprintp.fillna(0)
    return {
        'data': [
            go.Scatter(
                x=footprintp[footprintp['UN_region'] == i]['year'],
                y=footprintp[footprintp['UN_region'] == i][value],
                text=footprintp[footprintp['UN_region'] == i]['country'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=i
            ) for i in footprintp.UN_region.unique()
        ],
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'Year'},
            yaxis={'title': 'Ecological Footprint Consumption (GHA)' if value == 'total' else 'Carbon Emission (GHA)'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
            title='Ecological Footprint Total Consumption from 1960-2014 (Clean)' if value == 'total' else 'Carbon Emission from 1960-2014 (Clean)',
            hovermode='closest'
        )
    }


@app.callback(Output('preprocessed-info-container', 'children'),
              [Input('preprocessed-info-dropdown', 'value')])
def update_info(value):
    data = pd.read_csv('NFA_2018.csv')
    footprintp = data[data.record == 'EFConsTotGHA']
    footprintp = footprintp[
        ['country', 'ISO alpha-3 code', 'UN_region', 'UN_subregion', 'year', 'record', 'crop_land', 'grazing_land',
            'forest_land', 'fishing_ground', 'built_up_land', 'population', 'carbon', 'total',
            'Percapita GDP (2010 USD)']]

    # nan missing data
    footprintp = footprintp.replace(0, np.NaN)
    # drop rows with no data and data from world
    footprintp = footprintp[footprintp.crop_land.notna()]
    footprintp = footprintp[footprintp.country != 'World']

    # fill nan with 0
    footprintp = footprintp.fillna(0)
    if value == 'observations':
        return footprintp.size
    elif value == 'features':
        return len(footprintp.columns)




###PAGE 3
page_3_layout = html.Div([
    html.H3('Linear Regression Model'),
    dcc.Dropdown(
        id='choose-var-dropdown',
        options=[
            {'label': 'Crop Land vs Carbon', 'value': 'crop'},
            {'label': 'Grazing Land vs Carbon', 'value': 'grazing'},
            {'label': 'Forest Land vs Carbon', 'value': 'forest'},
            {'label': 'Fishing Ground vs Carbon', 'value': 'fishing'},
            {'label': 'Built-Up Land vs Carbon', 'value': 'built'}
        ],
        value='crop'
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'R-Squared', 'value': 'r2'},
            {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
            {'label': 'Intercept', 'value': 'intercept'},
            {'label': 'Coefficients', 'value': 'coef'}
        ],
        value='r2'
    ),
    html.Div(id='model-info'),
    dcc.Graph(id='fit-data'),
    dcc.Link('Go back to Explore Data', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])



@app.callback(dash.dependencies.Output('fit-data','figure'),
              [dash.dependencies.Input('choose-var-dropdown', 'value')])
def fit(value):
    datasetm = pd.read_csv("NFA_2018.csv", delimiter=",")
    # subsetting observations from records EFConsTotGHA
    footprintm = datasetm[datasetm.record == 'EFConsTotGHA']  # create new dataframe
    # rearrange columns
    footprintm = footprintm[
        ['country', 'ISO alpha-3 code', 'UN_region', 'UN_subregion', 'year', 'record', 'crop_land', 'grazing_land',
         'forest_land', 'fishing_ground', 'built_up_land', 'population', 'carbon', 'total', 'Percapita GDP (2010 USD)']]
    # nan missing data
    footprintm = footprintm.replace(0, np.NaN)
    # drop rows with no data and data from world
    footprintm = footprintm[footprintm.crop_land.notna()]
    footprintm = footprintm[footprintm.country != 'World']
    # fill nan with 0
    footprintm = footprintm.fillna(0)
    # split dataset into X and Y
    values = footprintm.values
    if value == 'crop':
        var = 6
    elif value == 'grazing':
        var = 7
    elif value == 'forest':
        var = 8
    elif value == 'fishing':
        var = 9
    else:
        var = 10
    X = values[:, var]
    Y = values[:, 12]
    # train test split 70/30
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=50)
    # make into dataframes
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    # impute missing value with mean
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    # impute training set
    imp_x_train = imputer.fit_transform(x_train)
    imp_x_test = imputer.transform(x_test)
    imp_y_train = imputer.fit_transform(y_train)
    imp_y_test = imputer.transform(y_test)
    # scale data
    scaler = MinMaxScaler()
    scaled_x_train = scaler.fit_transform(imp_x_train)
    scaled_x_test = scaler.transform(imp_x_test)
    scaled_y_train = scaler.fit_transform(imp_y_train)
    scaled_y_test = scaler.transform(imp_y_test)
    #convert to dataframe
    scaled_x_train = pd.DataFrame(scaled_x_train)
    scaled_x_test = pd.DataFrame(scaled_x_test)
    scaled_y_train = pd.DataFrame(scaled_y_train)
    scaled_y_test = pd.DataFrame(scaled_y_test)
    # train with linear regression model using training sets
    model = LinearRegression(normalize=True)
    model.fit(scaled_x_train, scaled_y_train)
    carbon_y_pred = model.predict(scaled_x_test)
    return {
        'data': [
            go.Scatter(
                x=scaled_x_train,
                y=scaled_y_train,
                name=' Training Data',
                mode='markers',
                opacity=0.7
            ),
            go.Scatter(
                x=scaled_x_test,
                y=scaled_y_test,
                name='Test Data',
                mode='markers',
                opacity=0.7
            ),
            go.Scatter(
                x=scaled_x_test,
                y=carbon_y_pred,
                name='Prediction',
                mode='lines'

            )

        ],
        'layout': {
            'margin': {
                'l': 40,  # left margin, in px
                'r': 40,  # right margin, in px
                't': 40,  # top margin, in px
                'b': 80,  # bottom margin, in px
            },

        }
    }









    











####}links
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)