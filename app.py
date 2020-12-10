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
# Import the libraries
import plotly.figure_factory as ff
from plotly.offline import iplot
import math

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

##Reading the datasets
df = pd.read_csv(r"./Data_Files/risk_output.csv")
df1=pd.read_csv(r"./Data_Files/Comparison Analysis Outputs.csv")

##Mean and Percentile value Calculation for Stick Lines
Annualized_loss=df['Annualized Risk ($)']
# loss_ML=statistics.mode(Annualized_loss) 
loss_ML=32000 ##Hard coding here since Mode Calculation is wrong in the previous line due to binning and decimals.
dflen=len(Annualized_loss) ## For Y axis co-ordinate


loss_50th=Annualized_loss.median()
loss_10th=Annualized_loss.quantile(0.10)
loss_90th=Annualized_loss.quantile(0.90)

##Mean Calculation for Fig 0 (Approach Comaprison Plots)
loss_approach_a=df1[" approach_a "]
loss_a=statistics.median(loss_approach_a)
cnt=len(loss_approach_a)

loss_approach_b=df1[" approach_b "]
loss_b=statistics.median(loss_approach_b)

##Median and Percentile Calculations for Fig 0 (Approach Comaprison Plots)
loss_a_90th=loss_approach_a.quantile(0.9)
loss_a_10th=loss_approach_a.quantile(0.1)

loss_b_90th=loss_approach_b.quantile(0.9)
loss_b_10th=loss_approach_b.quantile(0.1)

##Defining Figures using Graph Objects
fig = go.Figure() ##First Figure


fig_A= go.Figure() ##Second Figure (Approach A Plot)
fig_B= go.Figure() ##Third Figure (Approach B Plot)

fig0=go.Figure()   ##Fourth Plot (Comparison Plot both A & B) 
fig1 = make_subplots() ## Fifth Plot (Top Risks)
fig3=go.Figure()    ##Sixth Plot (Boxplots)
fig4=go.Figure()    ##Seventh Plot (All Histograms)
fig5=px.scatter(x=df['Loss Event Frequency'],y=df['Annualized Risk ($)'],trendline='ols',trendline_color_override="red")
##Scater Plot Final Figure


##Defining the Layouts and Axis Labels before plotting the Graph

fig.update_layout(
    {
'plot_bgcolor': 'white',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},
# plot_bgcolor='gray'
title={
        'text': "How much Risk do We Have (Annualized Risk)?",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},shapes=[
        # Phase 1 & 2
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=loss_10th,
            y0=0,
            x1=loss_90th,
            y1=1,
            fillcolor="lightgray",
            opacity=0.5,
            layer="below",
            line_width=0,
        )],
        # xaxis_title_text='Annualized Risk ($)', # xaxis label
    yaxis_title_text='Simulation Distribution',
    xaxis=dict(
        title="price",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    )
     ,plot_bgcolor='white'
    )

fig_A.update_layout(
    {
'plot_bgcolor': 'white',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},

title={
        'text': "Histogram and Cumulative Probability for Approach A",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},shapes=[
        # Phase 1 & 2
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="0",
            y0=0,
            x1="1363000",
            y1=1,
            fillcolor="lightgray",
            opacity=0.5,
            layer="below",
            line_width=0,
        )],
        xaxis=dict(
        title="Annualized Risk ($)",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ),
    yaxis_title_text='Probability (%)')

fig_B.update_layout(
    {
'plot_bgcolor': 'white',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},shapes=[
        # Phase 1 & 2
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0="0",
            y0=0,
            x1="6499999",
            y1=1,
            fillcolor="lightgray",
            opacity=0.5,
            layer="below",
            line_width=0,
        )],
# plot_bgcolor='gray'
title={
        'text': "Histogram and Cumulative Probability for Approach B",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(
        title="Annualized Risk ($)",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ), # xaxis label
    yaxis_title_text='Probability (%)',)

fig0.update_layout(title={
        'text': "Could We Reduce Risk ?",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},shapes=[
        # Phase 1 & 2
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=loss_a_90th,
            y0=0,
            x1=loss_b_90th,
            y1=1,
            fillcolor="lightgray",
            opacity=0.5,
            layer="below",
            line_width=0,
        )],xaxis=dict(
        title="Annualized Risk ($)",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ),
    yaxis_title_text='Probability %',plot_bgcolor='white'
)

fig1.update_layout(title={
        'text': "What are our Top Risks ?",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(
          title='Risk Types',
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False # Removes Y-axis grid lines    
    ), # xaxis label
    yaxis_title_text='Loss Value ($)', # yaxis label
    plot_bgcolor='white'
)


fig3.update_layout(title={
        'text': 'Secondary Response Comparisons (Boxplots)',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    yaxis=dict(
        title="Number of Occurences",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ),xaxis=dict(
         
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ),
    plot_bgcolor='white' # yaxis label
)

fig4.update_layout(title={
        'text': 'Secondary Response Comparisons (Histograms)',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis=dict(
          title='Loss Exposure ($)',
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    ),
    yaxis_title_text='Number of Occurences',  plot_bgcolor='white'# yaxis label
)


fig5.update_layout(xaxis_title_text='Loss Event Frequency', # xaxis label
    yaxis_title_text='Annualized Risk ($)', plot_bgcolor='white',# yaxis label
    title={
        'text': "Correlation Between Loss Frequency and Annualized Loss",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)

##Plotting the Objects
##Fig Begins


fig.add_trace(go.Histogram(x=df['Annualized Risk ($)']))

##Plotting the Percentile Sticks using calculated Co-ordinates

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
fig.add_shape( ##Mode Line
        go.layout.Shape(type='line', xref='x',
                        x0=loss_ML, y0=0,x1=loss_ML,y1=650, line=dict(
        color="red",
        width=2,
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
fig.add_annotation(x=loss_ML,y=650,
            text="   ML",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=loss_90th,y=650,
            text="90th Percentile",
            showarrow=False,
            yshift=10)
fig.add_annotation(x=120000,y=550,
            text="80% chance that our loss Exposure would be between " +str("{:,}".format(math.trunc(loss_10th)))+" $ and "+str("{:,}".format(math.trunc(loss_90th)))+" $",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
        
        
fig.add_annotation(
    x=120000,
    y=450,
    text='<b><i>Most Likely Loss Value we could expect is '+str("{:,}".format(math.trunc(loss_ML)))+' $</i></b>',
    ax=0.5,
    ay=2,
    arrowhead=2,
)
fig.add_annotation(
    x=120000,
    y=350,
    text='<b><i>highest probable value would be <i></b>'+str("{:,}".format(math.trunc(loss_90th)))+' <b><i>$, in worst case it could be as high  as </i></b>'+str("{:,}".format(math.trunc(Annualized_loss.max())))+" <b><i>$</i><b>",
    ax=0.5,
    ay=2,
    arrowhead=2,
)

##fig Ends

##fig0 Begins
fig0.add_trace(go.Histogram(x=df1[" approach_a "],histnorm='percent',name='Approach-A'))
fig0.add_trace(go.Histogram(x=df1[" approach_b "],histnorm='percent',name='Approach-B'))


##Mean line Addition Approach A
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a, y0=0,x1=loss_a,y1=80, line=dict(
        color="blue",
        width=1,
    ))
)
##Mean line Addition Approach B
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b, y0=0,x1=loss_b,y1=80, line=dict(
        color="red",
        width=1,
    ))
)

##Median line Claculation Approach A
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a_90th, y0=0,x1=loss_a_90th,y1=80, line=dict(
        color="blue",
        width=1,
    ))
)
##Median line Claculation Approach B
fig0.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b_90th, y0=0,x1=loss_b_90th,y1=80, line=dict(
        color="red",
        width=1,
    ))
)

fig0.add_annotation(x=loss_a,y=80,
            text="50 th-A",
            showarrow=False,
            yshift=10)

fig0.add_annotation(x=loss_b,y=80,
            text="50th-B",
            showarrow=True,
            yshift=10)

fig0.add_annotation(x=loss_a_90th,y=80,
            text="90th - A",
            showarrow=True,
            yshift=10)

fig0.add_annotation(x=loss_b_90th,y=80,
            text="90th - B",
            showarrow=True,
            yshift=10)

fig0.add_annotation(x=14000000,y=70,
            text="Probability of any loss happening with Approach B is very less compared to A",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
fig0.add_annotation(
    x=14000000,
    y=60,
    text='<b><i>But in Max case (90 th Percentile) the loss due to B could be as high as  '+str("{:,}".format(math.trunc(loss_b_90th)))+" $",
    ax=0.5,
    ay=2,
    arrowhead=2,
)


##fig0 Ends
##fig1 Begins
fig1.add_trace(go.Bar(x=["Primary Response ($)","Reputation","Secondary Response","Fines and Judgements"],y=[df["Primary Response ($)"].median(),df["Reputation ($)"].median(),df["Secondary Response ($)"].median() ,df["Fines and Judgements ($)"].median()  ],marker_color='rgb(55, 83, 109)',text=["$"+str("{:,}".format(math.trunc(df["Primary Response ($)"].median()))),"$"+str("{:,}".format(math.trunc(df["Reputation ($)"].median()))),"$"+str("{:,}".format(math.trunc(df["Secondary Response ($)"].median()))) ,"$"+str("{:,}".format(math.trunc(df["Fines and Judgements ($)"].median())))],textposition='auto'),
              row=1, col=1)

fig1.add_annotation(x=2,y=40000,
            text="<b>Top Risks and its Expected loss value based on 50th Percentile values</b>",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
        
fig1.add_annotation(
    x=2,
    y=30000,
    text='<b><i>Reputation Losses Contributes most to the Annualized Risk out of the Secondary Responses</i></b>',
    ax=0.5,
    ay=2,
    arrowhead=2,
)
##fig1 Ends

##fig3 Begins
fig3.add_trace(go.Box(y=df["Reputation ($)"],name='Reputation'))
fig3.add_trace(go.Box(y=df["Secondary Response ($)"],name='Secondary Response'))
fig3.add_trace(go.Box(y=df["Fines and Judgements ($)"],name='Fines and Judgements'))
fig3.add_annotation(x=1.5,y=1750,
            text="Loss Exposure due to Reputation can range between " +str("{:,}".format(math.trunc(df["Reputation ($)"].quantile(0.10))))+" $ (10th) and "+str("{:,}".format(math.trunc(df["Reputation ($)"].quantile(0.90))))+" $ (90th)",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
fig3.add_annotation(
    x=1.5,
    y=1500,
    text='<b><i>Secondary Loss exposure ranges between '+str("{:,}".format(math.trunc(df["Secondary Response ($)"].quantile(0.10))))+" $ and "+str("{:,}".format(math.trunc(df["Secondary Response ($)"].quantile(0.90)))+" $"),
    ax=0.5,
    ay=2,
    arrowhead=2,
)
fig3.add_annotation(
    x=1.5,
    y=1250,
    text='<b><i>Fines and Judgements ranges between  '+str("{:,}".format(math.trunc(df["Fines and Judgements ($)"].quantile(0.10))))+" $ and "+str("{:,}".format(math.trunc(df["Fines and Judgements ($)"].quantile(0.90)))+" $"),
    ax=0.5,
    ay=2,
    arrowhead=2,
)
##fig3 Ends

##fig4 Begins
fig4.add_trace(go.Histogram(x=df['Secondary Response ($)'], name='Secondary Response'))
fig4.add_trace(go.Histogram(x=df['Fines and Judgements ($)'], name='Fines and Judgements'))
fig4.add_trace(go.Histogram(x=df['Reputation ($)'], name='Reputation'))
fig4.add_annotation(x=1200,y=750,
            text="Reputation loss is widely distributed compared to other secondary losses",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
fig4.add_annotation(
    x=1200,
    y=600,
    text='<b><i>Reputation Losses are destructive, which could hamper the capability of doing business</i></b>',
    ax=0.5,
    ay=2,
    arrowhead=2,
)
##fig4 Ends

##FIG_A Begins
binned = np.histogram([df1[" approach_a "]], bins=25)
plot_y = (np.cumsum(binned[0])/5000)*100

# Line
trace1 = go.Scatter(
    x=binned[1],
    y=plot_y,
    mode='lines',
    name="Cumulative Dist-A",
    hoverinfo='all',
    # line=dict(color = 'rgb(1255, 0, 0)'
    )

# go.Histogram(x=df1[" approach_a "], cumulative_enabled=True,histnorm='percent',name='Cumulative Freq Plot'

trace2 = go.Histogram(
    x=df1[" approach_a "],histnorm='percent',name='Approach_A'
   
    )

data = [trace1,trace2]



# Make figure


fig_A.add_trace(trace1)
fig_A.add_trace(trace2)
##FIG_A Ends

##FIG_B Begins

binned = np.histogram([df1[" approach_b "]], bins=25)
plot_y = (np.cumsum(binned[0])/5000)*100

# Line
trace1 = go.Scatter(
    x=binned[1],
    y=plot_y,
    mode='lines',
    name="Cumulative Dist-B",
    hoverinfo='all',
    # line=dict(color = 'rgb(1255, 0, 0)'
    )

trace2 = go.Histogram(
    x=df1[" approach_b "],histnorm='percent',name='Approach_B'
   
    )

data = [trace1,trace2]


# Make figure
fig_B.add_trace(trace1)
fig_B.add_trace(trace2)     
##FIG_B Ends

##Lines and Annotation Additions
##Median line Claculation Approach A (FIG_A)
fig_A.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a_90th, y0=0,x1=loss_a_90th,y1=90, line=dict(
        color="black",
        width=1,
    ))
)
fig_A.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a, y0=0,x1=loss_a,y1=90, line=dict(
        color="red",
        width=1,
    ))
)
fig_A.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_a_10th, y0=0,x1=loss_a_10th,y1=90, line=dict(
        color="black",
        width=1,
    ))
)


fig_A.add_annotation(x=loss_a_90th,y=90,
            text="90th Percentile",
            showarrow=True,
            yshift=10)
fig_A.add_annotation(x=loss_a_10th,y=90,
            text="10th Percentile",
            showarrow=True,
            yshift=10)
fig_A.add_annotation(x=3500000,y=80,
            text="80% chance that our loss Exposure would be between " +str("{:,}".format(math.trunc(loss_a_10th)))+" $ and "+str("{:,}".format(math.trunc(loss_a_90th)))+" $",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)

fig_A.add_annotation(
    x=3500000,
    y=60,
    text='<b><i>highest probable value would be <i></b>'+str("{:,}".format(math.trunc(loss_a_90th)))+' <b><i>$, in worst case it could be as high  as </i></b>'+str("{:,}".format(math.trunc(loss_approach_a.max())))+" <b><i>$</i><b>",
    ax=0.5,
    ay=2,
    arrowhead=2,
)
fig_A.add_annotation(
    x=3500000,
    y=40,   
    text='<b><i>There is a '+  str(100*(df1[" approach_a "].isin([0]).sum())/df1[" approach_a "].count())  + '% chance that the loss would be 0 $ </i><b>',
    ax=0.5,
    ay=2,
    arrowhead=2,
)


##Median line Claculation Approach B
fig_B.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b_90th, y0=0,x1=loss_b_90th,y1=90, line=dict(
        color="black",
        width=1,
    ))
)
fig_B.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b, y0=0,x1=loss_b,y1=90, line=dict(
        color="red",
        width=1,
    ))
)
fig_B.add_shape(
        go.layout.Shape(type='line', xref='x',
                        x0=loss_b_10th, y0=0,x1=loss_b,y1=90, line=dict(
        color="black",
        width=1,
    ))
)
fig_B.add_annotation(x=14000000,y=80,
             text="80% chance that our loss Exposure would be between " +str("{:,}".format(math.trunc(loss_b_10th)))+" $ and "+str("{:,}".format(math.trunc(loss_b_90th)))+" $",
            showarrow=False,
            yshift=10,font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
fig_B.add_annotation(
    x=14000000,
    y=65,
    text='<b><i>highest probable value would be <i></b>'+str("{:,}".format(math.trunc(loss_b_90th)))+' <b><i>$, in worst case it could be as high as </i></b>'+str("{:,}".format(math.trunc(loss_approach_b.max())))+"<b><i>$</i><b>",
    ax=0.5,
    ay=2,
    arrowhead=2,
)
fig_B.add_annotation(
    x=14000000,
    y=50,   
    text='<b><i>There is a '+  str(100*(df1[" approach_b "].isin([0]).sum())/df1[" approach_b "].count())  + '% chance that the loss would be 0 $ </i><b>',
    ax=0.5,
    ay=2,
    arrowhead=2,
)
fig_B.add_annotation(x=loss_b_90th,y=90,
            text="90th Percentile",
            showarrow=True,
            yshift=10)

fig_A.add_annotation(x=loss_a,y=90,
            text="50th Percentile",
            showarrow=True,
            yshift=10)
fig_B.add_annotation(x=loss_b_10th,y=90,
            text="10th Percentile",
            showarrow=True,
            yshift=10)

##FIG_A and FIG_B lines and annotation ends

###Summary of Simulation Table Plotting Starts

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

###Plotting the DataTable
fig_table = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Approach_A</b>','<b>Minimum</b>','<b>Most Likely</b>','<b>Average</b>','<b>Maximum</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      ['Loss Events Per Year', 'Loss Magnitude Per Event', '<b>Approach_B</b>', 'Loss Events Per Year', 'Loss Magnitude Per Event'],
      [0, '$'+str(df1[" approach_a "].min()),'' , 0, '$'+str(df1[" approach_b "].min())],
      [0, '$'+str(int(df1[" approach_a "].mode())),'' , 0, '$'+str(int(df1[" approach_b "].mode()))],
      [0, '$'+str(int(df1[" approach_a "].mean())) ,'' , 0, '$'+str(int(df1[" approach_b "].mean()))],
      [0, '$'+str(int(df1[" approach_a "].max())),'' , 0, '$'+str(int(df1[" approach_b "].max()))]],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
    align = ['left', 'center'],
    font = dict(color = 'darkslategray', size = 11)
    ))
])

fig_table.update_layout(
    {
'plot_bgcolor': 'white',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},

title={
        'text': "Summary of Simulation Results",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},xaxis_title_text='Annualized Risk ($)', # xaxis label
    yaxis_title_text='Loss Value ($)',)
colors={
    'background': '#111111',
    'text': '#7FDBFF'
 }

###Summary Table Plotting Ends



###Main Layout and Addition of plots
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
        id='Approacha',
        figure=fig_A
    ),

       dcc.Graph(
        id='Approachb',
        figure=fig_B
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
        id='Boxsecondary',
        figure=fig3
    ),
    dcc.Graph(
        id='Distribution',
        figure=fig4
    ),
        dcc.Graph(
        id='table',
        figure=fig_table
    ),
    dcc.Graph(
        id='Correlation',
        figure=fig5
    ),
], style={'width':'100%',"position":"absolute"})


if __name__ == '__main__':
    app.run_server(debug=True)