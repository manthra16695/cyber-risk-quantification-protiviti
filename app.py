# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import base64
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = pd.read_csv(r"./Data_Files/risk_output.csv")
# x=df["Annualized Risk ($)"].to_numpy()

fig = go.Figure()
fig0 = make_subplots()
fig1 = go.Figure()

fig2=px.scatter(x=df['Loss Event Frequency'],y=df['Annualized Risk ($)'],trendline='ols',trendline_color_override="red")

fig3=go.Figure()
fig4=go.Figure()
# np.random.seed(1)

# x = np.random.randn(1000)
# hist_data = [x]

# group_labels = ['distplot'] # name of the dataset

# fig5 = ff.create_distplot([df['Annualized Risk ($)']], group_labels,bin_size=0.2,curve_type='normal')



fig.update_layout(title_text='<b>How Much Risk Do We have ?</b>',xaxis_title_text='Annualized Risk ($)', # xaxis label
    yaxis_title_text='Number of Simulations')
fig0.update_layout(title_text='<b>What are our Top Risks?</b>',xaxis_title_text='Risk Types', # xaxis label
    yaxis_title_text='Loss Value ($)', # yaxis label
)

fig1.update_layout(title_text='<b>Secondary Response Comparisons</b>',xaxis_title_text='Dollar Value ($)', # xaxis label
    yaxis_title_text='Number of Simulations', # yaxis label
)
fig2.update_layout(title_text='<b>Correlation Between Loss Frequency and Annualized Loss</b>',xaxis_title_text='Loss Event Frequency', # xaxis label
    yaxis_title_text='Annualized Risk ($)', # yaxis label
)

fig3.update_layout(title_text='<b>BoxPlot Comparison of Annualized Loss and Primary Response</b>',
    yaxis_title_text='Loss Due to Risk ($)', # yaxis label
)

fig4.update_layout(title_text='<b>BoxPlot Comparison of Seconday Responses</b>',
    yaxis_title_text='Loss Due to Risk ($)', # yaxis label
)

# fig5.update_layout(title_text='<b>BoxPlot Comparison of Seconday Responses</b>',
#     yaxis_title_text='Loss Due to Risk ($)', # yaxis label
# )

fig.add_trace(go.Histogram(x=df['Annualized Risk ($)']))




fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=41015, y0=0,x1=41015,y1=650, line=dict(
        color="red",
        width=2
    ),name='Avg')
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=66576.15, y0=0,x1=66576.15,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=19618.57, y0=0,x1=19618.57,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=38069.51, y0=0,x1=38069.51,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_annotation(x=38069.51,y=650,
            text="50th",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=19618.57,y=650,
            text="10th Percentile",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=41015,y=650,
            text="   Avg",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=66576.15,y=650,
            text="90th Percentile",
            showarrow=False,
            yshift=10)
            
fig0.add_trace(go.Bar(x=["Primary Response ($)","Annualized Risk","Reputation","Secondary Response","Fines and Judgements"],y=[df["Primary Response ($)"].median(),df["Annualized Risk ($)"].median(),df["Reputation ($)"].median(),df["Secondary Response ($)"].median() ,df["Fines and Judgements ($)"].median()  ],marker_color='rgb(55, 83, 109)',text=[df["Primary Response ($)"].median(),df["Annualized Risk ($)"].median(),df["Reputation ($)"].median(),df["Secondary Response ($)"].median() ,df["Fines and Judgements ($)"].median() ],textposition='auto'),
              row=1, col=1)


fig1.add_trace(go.Histogram(x=df['Secondary Response ($)'], name='Secondary Response'))
fig1.add_trace(go.Histogram(x=df['Fines and Judgements ($)'], name='Fines and Judgements'))
fig1.add_trace(go.Histogram(x=df['Reputation ($)'], name='Reputation'))


fig3.add_trace(go.Box(y=df["Annualized Risk ($)"],name='Annualized Risk'))
fig3.add_trace(go.Box(y=df["Primary Response ($)"],name='Primary Response'))


fig4.add_trace(go.Box(y=df["Secondary Response ($)"],name='Secondary Response'))
fig4.add_trace(go.Box(y=df["Fines and Judgements ($)"],name='Fines and Judgements'))
fig4.add_trace(go.Box(y=df["Reputation ($)"],name='Reputation'))

# Reduce opacity to see both histograms
# fig.update_traces(opacity=0.75)
# fig.show()
colors={
    'background': '#111111',
    'text': '#7FDBFF'
 }

style={'backgroundColor': colors['background'], 'color': colors['text'], 'height':'100vh', 'width':'100%', 'height':'100%', 'top':'0px', 'left':'0px'}
image_filename = r"./assets/Protiviti.png" # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
     html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
    dcc.Graph(
        id='Annualized_Risk Distribution',
        figure=fig
    ),
    #    dcc.Graph(
    #     id='Distributio',
    #     figure=fig5
    # ),
       dcc.Graph(
        id='Annualized_Risk',
        figure=fig0
    ),
         dcc.Graph(
        id='Box',
        figure=fig3
    ),
         dcc.Graph(
        id='Boxsecondary',
        figure=fig4
    ),
    dcc.Graph(
        id='Distribution',
        figure=fig1
    ),
    dcc.Graph(
        id='Correlation',
        figure=fig2
    )
], style={'backgroundColor':'black'})


if __name__ == '__main__':
    app.run_server(debug=True)