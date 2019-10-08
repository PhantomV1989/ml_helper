import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

a = np.random.rand(10, 3)
at = ['a <br />y'] * 10

fig.add_trace(go.Scatter3d(
    x=a[:, 0],
    y=a[:, 1],
    z=a[:, 2],
    hovertext=at,
    hoverinfo='text',  # this means xzy info is removed from hover
    name="a",
    mode='markers',
    marker=dict(
        size=12,
        color='red',  # set color to an array/list of desired values
        opacity=0.8
    )
))

b = np.random.rand(10, 3)

fig.add_trace(go.Scatter3d(
    x=b[:, 0],
    y=b[:, 1],
    z=b[:, 2],
    name="b",
    mode='lines+markers',
    marker=dict(
        size=5,
        color='blue',  # set color to an array/list of desired values
        opacity=0.8
    ),
    line=dict(
        color='black',  # set color to an array/list of desired values
        width=4,
        dash='dot'  # dash
    ),
))

c = np.random.rand(15, 3)

fig.add_trace(go.Scatter3d(
    x=c[:, 0],
    y=c[:, 1],
    z=c[:, 2],
    name="c",
    mode='lines',
    line=dict(
        color='green',  # set color to an array/list of desired values
        width=5,
        dash='dash'  # dash
    ),
))
fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig.show()
