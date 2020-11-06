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
import statistics

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

##Reading the datasets
df = pd.read_csv(r"./Data_Files/risk_output.csv")
df1=pd.read_csv(r"./Data_Files/Comparison Analysis Outputs.csv")

##Defining Figures using Graph Objects
fig = go.Figure()
fig0=go.Figure()
fig1 = make_subplots()
fig2 = go.Figure()

fig3=go.Figure()
fig4=go.Figure()
fig5=px.scatter(x=df['Loss Event Frequency'],y=df['Annualized Risk ($)'],trendline='ols',trendline_color_override="red")


##Defining the Layouts and Axis Labels

fig.update_layout(title_text='<b>How Much Risk Do We have ?</b>',xaxis_title_text='Annualized Risk ($)', # xaxis label
    yaxis_title_text='Number of Simulations')

fig0.update_layout(title_text='<b>Comparison of Approaches</b>',
    yaxis_title_text='Loss Due to Risk ($)', # yaxis label
)

fig1.update_layout(title_text='<b>What are our Top Risks?</b>',xaxis_title_text='Risk Types', # xaxis label
    yaxis_title_text='Loss Value ($)', # yaxis label
)
fig2.update_layout(title_text='<b>BoxPlot Comparison of Annualized Loss and Primary Response</b>',
    yaxis_title_text='Loss Due to Risk ($)', # yaxis label
)

fig3.update_layout(title_text='<b>Secondary Response Comparisons</b>',xaxis_title_text='Dollar Value ($)', # xaxis label
    yaxis_title_text='Number of Simulations', # yaxis label
)

fig4.update_layout(title_text='<b>Histogram Comparison of Seconday Responses</b>',xaxis_title_text='Loss Due to Risk ($)',
    yaxis_title_text='Number of Simulations', # yaxis label
)

fig5.update_layout(title_text='<b>Correlation Between Loss Frequency and Annualized Loss</b>',xaxis_title_text='Loss Event Frequency', # xaxis label
    yaxis_title_text='Annualized Risk ($)', # yaxis label
)


##Plotting the Objects
##Fig Begins
fig.add_trace(go.Histogram(x=df['Annualized Risk ($)']))

##Mean and Percentile Calculation for Stick Lines
Annualized_loss=df['Annualized Risk ($)']
loss_avg=statistics.mean(Annualized_loss)
dflen=len(Annualized_loss) ## For Y axis co-ordinate


loss_50th=Annualized_loss.median()

loss_10th=Annualized_loss.quantile(0.10)
loss_90th=Annualized_loss.quantile(0.90)

##Plotting the Percentile Sticks using calculated Co-ordinates
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_avg, y0=0,x1=loss_avg,y1=650, line=dict(
        color="red",
        width=2
    ),name='Avg')
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_90th, y0=0,x1=loss_90th,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_10th, y0=0,x1=loss_10th,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_50th, y0=0,x1=loss_50th,y1=650, line=dict(
        color="black",
        width=1,
    ))
)
fig.add_annotation(x=loss_50th,y=650,
            text="50th",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=loss_10th,y=650,
            text="10th Percentile",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=loss_avg,y=650,
            text="   Avg",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=loss_90th,y=650,
            text="90th Percentile",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=140000,y=650,
            text="90 th - "+str(loss_90th)+"$",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=140000,y=600,
            text="Avg - "+str(loss_avg)+"$",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=140000,y=550,
            text="50 th - " +str(loss_50th)+"$",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=140000,y=500,
            text="10 th - "+str(loss_10th)+"$",
            showarrow=False,
            yshift=10)

##fig Ends

##fig0 Begins
fig0.add_trace(go.Histogram(x=df1[" approach_a "],name='Approach-A'))
fig0.add_trace(go.Histogram(x=df1[" approach_b "],name='Approach-B'))


##Mean Calculation for Fig 0
loss_approach_a=df1[" approach_a "]
loss_a=statistics.mean(loss_approach_a)
cnt=len(loss_approach_a)

loss_approach_b=df1[" approach_b "]
loss_b=statistics.mean(loss_approach_b)

##Median Calculation for Fig 0
med_loss_a=loss_approach_a.median()

med_loss_b=loss_approach_b.median()

##Mean line Addition Approach A
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a, y0=0,x1=loss_a,y1=cnt, line=dict(
        color="blue",
        width=1,
    ))
)
##Mean line Addition Approach B
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b, y0=0,x1=loss_b,y1=cnt, line=dict(
        color="red",
        width=1,
    ))
)

##Median line Claculation Approach A
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=med_loss_a, y0=0,x1=med_loss_a,y1=cnt, line=dict(
        color="blue",
        width=1,
    ))
)
##Median line Claculation Approach B
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=med_loss_b, y0=0,x1=med_loss_b,y1=cnt, line=dict(
        color="red",
        width=1,
    ))
)

fig0.add_annotation(x=loss_a,y=cnt,
            text="       Mean-A",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=loss_b,y=cnt,
            text="       Mean-B",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=med_loss_a,y=cnt,
            text="50th",
            showarrow=True,
            yshift=10)

fig0.add_annotation(x=med_loss_b,y=cnt,
            text="50th",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=12000000,y=cnt,
            text="Average A - "+str(loss_10th)+"$",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=12000000,y=cnt-500,
            text="Average B - "+str(loss_b)+"$",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=12000000,y=cnt-1000,
            text="50 th % A - "+str(med_loss_a)+"$",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=12000000,y=cnt-1500,
            text="50 th % B - "+str(med_loss_b)+"$",
            showarrow=False,
            yshift=10)

##fig0 Ends
##fig1 Begins
fig1.add_trace(go.Bar(x=["Primary Response ($)","Annualized Risk","Reputation","Secondary Response","Fines and Judgements"],y=[df["Primary Response ($)"].median(),df["Annualized Risk ($)"].median(),df["Reputation ($)"].median(),df["Secondary Response ($)"].median() ,df["Fines and Judgements ($)"].median()  ],marker_color='rgb(55, 83, 109)',text=[df["Primary Response ($)"].median(),df["Annualized Risk ($)"].median(),df["Reputation ($)"].median(),df["Secondary Response ($)"].median() ,df["Fines and Judgements ($)"].median() ],textposition='auto'),
              row=1, col=1)

##fig1 Ends
##fig2 Begins
fig2.add_trace(go.Box(y=df["Annualized Risk ($)"],name='Annualized Risk'))
fig2.add_trace(go.Box(y=df["Primary Response ($)"],name='Primary Response'))
##fig2 Ends
##fig3 Begins
fig3.add_trace(go.Box(y=df["Secondary Response ($)"],name='Secondary Response'))
fig3.add_trace(go.Box(y=df["Fines and Judgements ($)"],name='Fines and Judgements'))
fig3.add_trace(go.Box(y=df["Reputation ($)"],name='Reputation'))
##fig3 Ends

##fig4 Begins
fig4.add_trace(go.Histogram(x=df['Secondary Response ($)'], name='Secondary Response'))
fig4.add_trace(go.Histogram(x=df['Fines and Judgements ($)'], name='Fines and Judgements'))
fig4.add_trace(go.Histogram(x=df['Reputation ($)'], name='Reputation'))
##fig4 Ends


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

       dcc.Graph(
        id='Approach',
        figure=fig0
    ),
       dcc.Graph(
        id='Top_Risks',
        figure=fig1
    ),
         dcc.Graph(
        id='BoxPrimary',
        figure=fig2
    ),
         dcc.Graph(
        id='Boxsecondary',
        figure=fig3
    ),
    dcc.Graph(
        id='Distribution',
        figure=fig4
    ),
    dcc.Graph(
        id='Correlation',
        figure=fig5
    ),

], style={'width':'100%',"position":"absolute",'backgroundColor':'black'})


if __name__ == '__main__':
    app.run_server(debug=True)