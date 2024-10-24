import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sinusoidal_embedding(pos, dim, max_freq=10000):
    div_term = np.exp(np.arange(0, dim, step=2) * -(np.log(max_freq) / dim))
    angles = pos * div_term
    return np.concatenate([np.sin(angles), np.cos(angles)])


# Create the base figure with two subplots
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Sine Component", "Cosine Component"),
    horizontal_spacing=0,
)

# Add traces for sin and cos components
dim = 1024  # changes granularity
x_sin = np.arange(dim // 2)
x_cos = np.arange(dim // 2, dim)
initial_pos_embedding = sinusoidal_embedding(0, dim)
fig.add_trace(
    go.Scatter(
        x=x_sin,
        y=initial_pos_embedding[: dim // 2],
        name="Sin",
        line=dict(color="#228be6"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_cos,
        y=initial_pos_embedding[dim // 2 :],
        name="Cos",
        line=dict(color="#fa5252"),
    ),
    row=1,
    col=2,
)

# Create and add slider
steps = [
    dict(
        method="update",
        args=[
            {
                "y": [
                    initial_pos_embedding[: dim // 2],
                    initial_pos_embedding[dim // 2 :],
                ]
            },
        ],
        label=f"{0:.0f}",
    )
]
for exp in range(0, 5):  # This will cover positions from 1 to 10000
    for i in range(1, 10):
        pos = i * 10**exp
        embedding = sinusoidal_embedding(pos, dim)
        step = dict(
            method="update",
            args=[
                {"y": [embedding[: dim // 2], embedding[dim // 2 :]]},
            ],
            label=f"{pos:.0f}",
        )
        steps.append(step)

        if exp == 4 and i == 1:
            break  # only capture up to 10000

sliders = [
    dict(active=0, currentvalue={"prefix": "Position: "}, pad={"t": 100}, steps=steps)
]

fig.update_layout(
    sliders=sliders,
    autosize=True,
    height=350,
    width=600,
    font=dict(
        family="et-book, Palatino, 'Palatino Linotype', 'Palatino LT STD', 'Book Antiqua', Georgia, serif"
    ),
    paper_bgcolor="#fffff0",
    plot_bgcolor="#fffff0",
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=False,
)

fig.update_xaxes(title_text="Embedding Dimension", row=1, col=1, showgrid=False)
fig.update_xaxes(title_text="Embedding Dimension", row=1, col=2, showgrid=False)
fig.update_yaxes(title_text="Value", range=[-1, 1], row=1, col=1, showgrid=False)
fig.update_yaxes(
    title_text="", range=[-1, 1], showticklabels=False, row=1, col=2, showgrid=False
)

# Generate HTML with custom config
html_output = fig.to_html(
    full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
)

# Add custom classes to the Plotly output div and include a caption
modified_html = (
    '<p class="visualization-caption">Figure 1: Sinusoidal Position Encodings.</p>'
    + html_output.replace(
        'class="plotly-graph-div"', 'class="plotly-graph-div visualization-wrapper"'
    )
)

# Save the HTML to a file
with open("../../src/_includes/sinusoidal-visualization.html", "w") as f:
    f.write(modified_html)

print("HTML file 'sinusoidal-visualization.html' has been created.")
