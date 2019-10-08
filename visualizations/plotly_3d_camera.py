import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=["2017-01-01", "2017-02-10", "2017-03-20"],
    y=["A", "B", "C"],
    z=[1, 1000, 100000],
    name="z",
))

fig.update_layout(
    scene=go.layout.Scene(
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.96903462608, y=-1.09022831971, z=0.405345349304),
                    up=dict(x=0, y=0, z=1)
                    ),
        dragmode="turntable",
        xaxis=dict(title_text="", type="date"),
        yaxis=dict(title_text="", type="category"),
        zaxis=dict(title_text="", type="log"),
        annotations=[
            dict(showarrow=False, x="2017-01-01", y="A", z=0, text="Point 1", xanchor="left", xshift=10, opacity=0.7),
            dict(x="2017-02-10", y="B", z=4, text="Point 2", textangle=0, ax=0, ay=-75,
                 font=dict(color="black", size=12), arrowcolor="black", arrowsize=3, arrowwidth=1, arrowhead=1),
            dict(x="2017-03-20", y="C", z=5, ax=50, ay=0, text="Point 3",
                 arrowhead=1, xanchor="left", yanchor="bottom")]),
    xaxis=dict(title_text="x"),
    yaxis=dict(title_text="y")
)

fig.show()
