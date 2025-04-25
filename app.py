import base64
import os
import time
from functools import lru_cache

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

# Initialize the Dash app with external styles
app = Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap'
])
server = app.server

# Configuration
classroom_config = {
    "teacher_area": (0.3, 0.1, 0.7, 0.3),
    "board_area": (0.1, 0.1, 0.9, 0.3),
    "student_areas": [
        (0.1, 0.4, 0.3, 0.8),
        (0.3, 0.4, 0.5, 0.8),
        (0.5, 0.4, 0.7, 0.8),
        (0.7, 0.4, 0.9, 0.8)
    ]
}

# Global Variables
analysis_mode = "upload"  # Fixed to upload
attention_data = None  # Store latest /analyze_frame results
@lru_cache(maxsize=128)
def analyze_frame_cache(key):
    # Split the key to get upload_id and frame_idx
    upload_id, frame_idx = key.split('_')
    frame_idx = int(frame_idx)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Fetching data for cache_key: {key}")
            response = requests.post(
                API_ANALYZE_FRAME,
                json={'upload_id': upload_id, 'frame_idx': frame_idx},
                
                verify=False
            )
            print(f"/analyze_frame status: {response.status_code}, response: {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Non-200 status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for cache_key {key}: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Max retries exceeded for cache_key {key}")
                return None
            time.sleep(1)  # Wait before retrying
    return None

# API Endpoints
NGROK_URL = "http://localhost:5000"
API_UPLOAD_FRAME = f"{NGROK_URL}/upload_frame"
API_GET_FRAME = f"{NGROK_URL}/get_frame"
API_ANALYZE_FRAME = f"{NGROK_URL}/analyze_frame"
API_ANALYZE_TEMPORAL = f"{NGROK_URL}/analyze_temporal"
API_VISUALIZE_FRAME = f"{NGROK_URL}/visualize_frame"

# Custom CSS styles
custom_styles = {
    'header': {
        'background': 'linear-gradient(135deg, #3498db 0%, #2c3e50 100%)',
        'color': 'white',
        'padding': '1.5rem',
        'borderRadius': '0 0 10px 10px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
        'marginBottom': '2rem'
    },
    'card': {
        'background': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
        'padding': '1.5rem',
        'marginBottom': '1.5rem',
        'border': '1px solid rgba(0,0,0,0.05)'
    },
    'button': {
        'primary': {
            'background': 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(52, 152, 219, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'danger': {
            'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(231, 76, 60, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'success': {
            'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(46, 204, 113, 0.3)',
            'transition': 'all 0.3s ease'
        },
        'secondary': {
            'background': 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(149, 165, 166, 0.3)',
            'transition': 'all 0.3s ease'
        }
    },
    'indicator': {
        'good': {
            'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        },
        'warning': {
            'background': 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        },
        'danger': {
            'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
            'color': 'white',
            'padding': '0.5rem 1rem',
            'borderRadius': '50px',
            'fontWeight': '600',
            'display': 'inline-block'
        }
    }
}

# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("MS-GESCAM Classroom Analytics", style={
                'marginBottom': '0.5rem',
                'fontWeight': '700',
                'fontSize': '2.2rem'
            }),
            html.P("Multi-Stream Gaze Estimation for Classroom Attention Measurement", style={
                'marginTop': '0',
                'fontSize': '1.1rem',
                'opacity': '0.9'
            }),
            html.Div([
                html.Span([
                    html.I(className="fas fa-book", style={'marginRight': '8px'}),
                    "18-799-RW Applied Computer Vision"
                ], style={
                    'backgroundColor': 'rgba(255,255,255,0.2)',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '50px',
                    'fontSize': '0.9rem'
                })
            ], style={'marginTop': '1rem'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto'})
    ], style=custom_styles['header']),
    
    # Main content
    html.Div([
        # Left sidebar
        html.Div([
            # Analysis Mode Card
            html.Div([
                html.H4("Analysis Mode", style={
                    'marginBottom': '1rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                dcc.RadioItems(
                    id='analysis-mode',
                    options=[
                        {
                            'label': html.Span([
                                html.I(className="fas fa-video", style={'marginRight': '10px'}),
                                "Real-time Monitoring"
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                            'value': 'realtime',
                            'disabled': True
                        },
                        {
                            'label': html.Span([
                                html.I(className="fas fa-upload", style={'marginRight': '10px'}),
                                "Upload Recording"
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                            'value': 'upload'
                        }
                    ],
                    value='upload',
                    labelStyle={'display': 'block', 'marginBottom': '15px'},
                    inputStyle={'marginRight': '10px'}
                )
            ], style=custom_styles['card']),
            
            # Upload Section
            html.Div([
                dcc.Upload(
                    id='upload-video',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '2rem', 'marginBottom': '1rem'}),
                        html.P("Drag and Drop or Click to Select Image/Video (PNG/JPEG/MP4)")
                    ], style={'textAlign': 'center', 'padding': '2rem'}),
                    style={
                        'width': '100%',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '8px',
                        'textAlign': 'center',
                        'marginBottom': '1rem',
                        'cursor': 'pointer',
                        'borderColor': '#3498db',
                        'background': 'rgba(52, 152, 219, 0.05)'
                    },
                    multiple=False,
                    accept='image/png,image/jpeg,video/mp4'
                ),
                html.Div(id='upload-status', style={
                    'color': '#2c3e50',
                    'textAlign': 'center',
                    'padding': '0.5rem'
                })
            ], style=custom_styles['card']),
            
            # Overall Attention Card
            html.Div([
                html.H4("Overall Attention", style={
                    'marginBottom': '1rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                html.Div(id='overall-attention-score', children="N/A", style={
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '1rem',
                    'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    'padding': '1.5rem',
                    'borderRadius': '10px',
                    'boxShadow': 'inset 0 4px 15px rgba(0,0,0,0.05)'
                }),
                dcc.Graph(
                    id='student-scores-chart',
                    style={'height': '250px'},
                    config={'displayModeBar': False}
                )
            ], style=custom_styles['card'])
        ], className="three columns", style={'paddingRight': '20px'}),
        
        # Main content area
        html.Div([
            # Video Display Card
            html.Div([
                html.Div([
                    html.H3(id='view-title', children="Lecture Recording Analysis", style={
                        'marginBottom': '0.5rem',
                        'color': '#2c3e50'
                    }),
                    html.P("Upload an image or video for classroom analysis", style={
                        'marginTop': '0',
                        'color': '#7f8c8d',
                        'marginBottom': '1.5rem'
                    })
                ], style={'marginBottom': '1rem'}),
                
                html.Div([
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-eye", style={'marginRight': '8px'}),
                            "Raw Feed"
                        ]),
                        id='display-mode-raw',
                        n_clicks=0,
                        style={**custom_styles['button']['secondary'], 'marginRight': '10px'}
                    ),
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-fire", style={'marginRight': '8px'}),
                            "Heatmap"
                        ]),
                        id='display-mode-heatmap',
                        n_clicks=0,
                        style={**custom_styles['button']['danger'], 'marginRight': '10px'}
                    ),
                    html.Button(
                        html.Span([
                            html.I(className="fas fa-chart-line", style={'marginRight': '8px'}),
                            "Engagement"
                        ]),
                        id='display-mode-engagement',
                        n_clicks=0,
                        style=custom_styles['button']['success']
                    )
                ], style={'marginBottom': '1.5rem'}),
                
                # Frame Selector Slider
                html.Div([
                    html.Label("Select Frame:", style={'marginRight': '10px'}),
                    dcc.Slider(
                        id='frame-selector',
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': '1rem'}),
                
                html.Div(
                    id='video-display',
                    children=html.P("Upload an image or video to view analysis", style={'textAlign': 'center', 'padding': '2rem'}),
                    style={
                        'borderRadius': '10px',
                        'overflow': 'hidden',
                        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
                        'height': '400px',
                        'background': '#f8f9fa',
                        'border': '1px solid rgba(0,0,0,0.05)'
                    }
                )
            ], style={
                **custom_styles['card'],
                'padding': '2rem'
            }),
            
            # Engagement Alert
            html.Div(id='engagement-alert', style={'marginBottom': '1.5rem'}),
            
            # Charts Row
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Engagement Overview", style={
                            'marginBottom': '1rem',
                            'color': '#2c3e50'
                        }),
                        dcc.Graph(
                            id='engagement-chart', 
                            style={'height': '250px'},
                            config={'displayModeBar': False}
                        )
                    ], style=custom_styles['card'])
                ], className="six columns"),
                
                html.Div([
                    html.Div([
                        html.H4("Attention Trend", style={
                            'marginBottom': '1rem',
                            'color': '#2c3e50'
                        }),
                        dcc.Graph(
                            id='attention-trend', 
                            style={'height': '250px'},
                            config={'displayModeBar': False}
                        )
                    ], style=custom_styles['card'])
                ], className="six columns")
            ], className="row")
        ], className="six columns", style={'paddingRight': '20px'}),
        
        # Right sidebar
        html.Div([
            html.Div([
                html.H4("Student Analytics", style={
                    'marginBottom': '1.5rem',
                    'color': '#2c3e50',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
                dcc.Dropdown(
                    id='student-selector',
                    options=[],
                    value=None,
                    clearable=False,
                    style={'marginBottom': '1.5rem'}
                ),
                html.Div(id='student-details', children="No student data available")
            ], style=custom_styles['card'])
        ], className="three columns")
    ], className="row", style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    # Hidden stores
    dcc.Store(id='upload-data', data={'upload_id': None, 'frame_count': 0}),
    dcc.Store(id='historical-data', data={'timestamps': [], 'scores': []}),
    dcc.Store(id='session-data'),
    dcc.Store(id='display-mode', data='raw'),
    dcc.Store(id='upload-timestamp', data=None),
    dcc.Interval(id='live-update', interval=1000, disabled=True),
    html.Div(id='dummy-output', style={'display': 'none'})
], style={
    'fontFamily': '"Open Sans", sans-serif',
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
    'paddingBottom': '2rem'
})

# Visualization functions
def generate_live_view(upload_id, frame_idx):
    try:
        response = requests.post(
            API_GET_FRAME,
            json={'upload_id': upload_id, 'frame_idx': frame_idx},
            
            verify=False
        )
        print(f"/get_frame response: {response.json()}")
        if response.status_code == 200:
            frame_base64 = response.json().get('original_frame')
            return html.Img(
                src=f"data:image/jpeg;base64,{frame_base64}",
                style={'width': '100%', 'height': '100%', 'objectFit': 'contain'},
                key=f"raw-{frame_idx}"
            )
        return html.P(f"Failed to load frame: {response.json().get('error', 'Unknown error')}", style={'textAlign': 'center', 'padding': '2rem'})
    except Exception as e:
        return html.P(f"Error loading frame: {str(e)}", style={'textAlign': 'center', 'padding': '2rem'})

def generate_heatmap_view(upload_id, frame_idx):
    try:
        response = requests.post(
            API_VISUALIZE_FRAME,
            json={'upload_id': upload_id, 'frame_idx': frame_idx, 'vis_type': 'heatmap'},
            verify=False
        )
        print(f"/visualize_frame (heatmap) response: {response.json()}")
        if response.status_code == 200:
            vis_data = response.json().get('vis_image')
            return html.P(f"Heatmap Path: {vis_data}", style={'textAlign': 'center', 'padding': '2rem'})
        return html.P(f"Heatmap failed: {response.json().get('error', 'Unknown error')}", style={'textAlign': 'center', 'padding': '2rem'})
    except Exception as e:
        return html.P(f"Heatmap error: {str(e)}", style={'textAlign': 'center', 'padding': '2rem'})

def generate_engagement_view(upload_id, frame_idx):
    try:
        response = requests.post(
            API_VISUALIZE_FRAME,
            json={'upload_id': upload_id, 'frame_idx': frame_idx, 'vis_type': 'engagement'},
            
            verify=False
        )
        print(f"/visualize_frame (engagement) response: {response.json()}")
        if response.status_code == 200:
            vis_data = response.json().get('vis_image')
            return html.P(f"Engagement: {vis_data}", style={'textAlign': 'center', 'padding': '2rem'})
        return html.P(f"Engagement view failed: {response.json().get('error', 'Unknown error')}", style={'textAlign': 'center', 'padding': '2rem'})
    except Exception as e:
        return html.P(f"Engagement view error: {str(e)}", style={'textAlign': 'center', 'padding': '2rem'})

def generate_engagement_figure(score):
    engaged = score if 0 <= score <= 1 else 0
    not_engaged = 1 - engaged
    fig = px.pie(
        values=[engaged, not_engaged],
        names=["Engaged", "Not Engaged"],
        hole=0.4,
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': '"Open Sans", sans-serif'}
    )
    return fig

def generate_trend_figure(upload_id, frame_count):
    if not upload_id:
        fig = px.line()
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text="No data yet",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )]
        )
        return fig
    
    try:
        # Fetch temporal data
        response = requests.post(
            API_ANALYZE_TEMPORAL,
            json={'upload_id': upload_id},
            
            verify=False
        )
        print(f"/analyze_temporal response: {response.json()}")
        if response.status_code != 200:
            raise Exception("Failed to fetch temporal data")
        
        trend_data = response.json().get('trend', [])
        if not trend_data:
            raise Exception("No trend data available")
        
        # Fetch per-frame student engagement
        frames_data = []
        for frame_idx in range(frame_count):
            cache_key = f"{upload_id}_{frame_idx}"
            frame_data = analyze_frame_cache(cache_key)
            if frame_data:
                frames_data.append({
                    'frame_idx': frame_idx,
                    'mean_attention': frame_data.get('frame_stats', {}).get('mean_attention', 0.0),
                    'individual_scores': frame_data.get('individual_scores', [])
                })
            else:
                frames_data.append({
                    'frame_idx': frame_idx,
                    'mean_attention': 0.0,
                    'individual_scores': []
                })
        
        # Create traces for mean attention
        fig = go.Figure()
        mean_trace = go.Scatter(
            x=[item['frame_idx'] for item in frames_data],
            y=[item['mean_attention'] for item in frames_data],
            mode='lines+markers',
            name='Mean Attention',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        )
        fig.add_trace(mean_trace)
        
        # Add individual person traces (optional, can be toggled in legend)
        person_ids = set()
        for frame in frames_data:
            for score in frame['individual_scores']:
                person_ids.add(score['person_idx'])
        
        for person_id in person_ids:
            person_scores = []
            for frame in frames_data:
                score = next((s['attention_score'] for s in frame['individual_scores'] if s['person_idx'] == person_id), 0.0)
                person_scores.append(score)
            fig.add_trace(go.Scatter(
                x=[item['frame_idx'] for item in frames_data],
                y=person_scores,
                mode='lines',
                name=f"Person ID: {person_id}",
                line=dict(width=1, dash='dash'),
                visible='legendonly'  # Hidden by default, can be toggled in legend
            ))
        
        # Add slider for frame navigation
        steps = []
        for frame_idx in range(frame_count):
            # Prepare x and y data for all traces
            x_data = [frame_idx]  # For mean attention trace
            y_data = [frames_data[frame_idx]['mean_attention']]  # For mean attention trace
            
            # Add x and y data for each person trace
            person_x_data = [frame_idx] * len(person_ids)
            person_y_data = [
                next((s['attention_score'] for s in frames_data[frame_idx]['individual_scores'] if s['person_idx'] == pid), 0.0)
                for pid in person_ids
            ]
            
            step = dict(
                method="update",
                args=[
                    {
                        "x": [x_data] + [person_x_data] * len(person_ids),  # x data for mean trace + each person trace
                        "y": [y_data] + [[score] for score in person_y_data]  # y data for mean trace + each person trace
                    },
                    {"title": f"Attention Trend - Frame {frame_idx}"}
                ],
                label=str(frame_idx)
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                range=[0, 1],
                gridcolor='rgba(0,0,0,0.05)',
                zeroline=False,
                title="Attention Score"
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                title="Frame"
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': '"Open Sans", sans-serif'},
            sliders=sliders,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        return fig
    except Exception as e:
        fig = px.line()
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text=f"Trend data unavailable: {str(e)}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )]
        )
        return fig

def generate_alert(score):
    if score < 0.4:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '10px'}),
            "Low Engagement! Consider changing activities."
        ], style={
            **custom_styles['indicator']['danger'],
            'padding': '1rem',
            'width': '100%',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
    elif score < 0.6:
        return html.Div([
            html.I(className="fas fa-exclamation-circle", style={'marginRight': '10px'}),
            "Moderate Engagement - Some students may need support"
        ], style={
            **custom_styles['indicator']['warning'],
            'padding': '1rem',
            'width': '100%',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
    return html.Div([
        html.I(className="fas fa-check-circle", style={'marginRight': '10px'}),
        "Good Engagement Level Maintained"
    ], style={
        **custom_styles['indicator']['good'],
        'padding': '1rem',
        'width': '100%',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    })

# Callbacks
@app.callback(
    [Output('upload-status', 'children'),
     Output('view-title', 'children'),
     Output('live-update', 'disabled'),
     Output('live-update', 'interval'),
     Output('upload-data', 'data'),
     Output('frame-selector', 'max'),
     Output('frame-selector', 'value'),
     Output('upload-timestamp', 'data')],
    [Input('upload-video', 'contents')],
    [State('upload-video', 'filename'),
     State('analysis-mode', 'value')]
)
def handle_upload(upload_contents, upload_filename, mode):
    global attention_data
    upload_status = ""
    
    if upload_contents is None:
        return "", "Lecture Recording Analysis", True, 1000, {'upload_id': None, 'frame_count': 0}, 0, 0, None
    
    try:
        # Decode the uploaded file
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Determine content type based on filename extension
        file_ext = os.path.splitext(upload_filename)[1].lower()
        if file_ext in ['.png', '.jpeg', '.jpg']:
            mime_type = 'image/png' if file_ext == '.png' else 'image/jpeg'
        elif file_ext in ['.mp4']:
            mime_type = 'video/mp4'
        else:
            upload_status = "Unsupported file type. Please upload PNG, JPEG, or MP4."
            return upload_status, "Lecture Recording Analysis", True, 1000, {'upload_id': None, 'frame_count': 0}, 0, 0, None
        
        # Upload to /upload_frame with retries
        files = {'file': (upload_filename, decoded, mime_type)}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Uploading to {API_UPLOAD_FRAME}")
                response = requests.post(
                    API_UPLOAD_FRAME,
                    files=files,
                    verify=False
                )
                print(f"Upload response status: {response.status_code}, body: {response.text}")
                break
            except requests.exceptions.RequestException as e:
                print(f"Upload attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
        
        if response.status_code == 200:
            response_json = response.json()
            upload_id = response_json.get('upload_id')
            frame_count = response_json.get('frame_count', 1)
            upload_status = f"Uploaded {upload_filename}; Frames: {frame_count}"
            # Clear cache on new upload
            try:
                analyze_frame_cache.cache_clear()
                print("Cache cleared successfully")
            except Exception as e:
                print(f"Failed to clear cache: {str(e)}")
            # Disable live updates for single-frame uploads to prevent excessive CLI logs
            interval = 3600000 if frame_count == 1 else 1000  # 1 hour for single frame, 1 second otherwise
            return upload_status, "Lecture Recording Analysis", frame_count == 1, interval, {'upload_id': upload_id, 'frame_count': frame_count}, frame_count - 1, 0, str(time.time())
        else:
            error_msg = response.json().get('error', 'Unknown error')
            upload_status = f"Upload failed: {error_msg}"
            return upload_status, "Lecture Recording Analysis", True, 1000, {'upload_id': None, 'frame_count': 0}, 0, 0, None
    except Exception as e:
        upload_status = f"Error: {str(e)}"
        return upload_status, "Lecture Recording Analysis", True, 1000, {'upload_id': None, 'frame_count': 0}, 0, 0, None

@app.callback(
    Output('display-mode', 'data'),
    [Input('display-mode-raw', 'n_clicks'),
     Input('display-mode-heatmap', 'n_clicks'),
     Input('display-mode-engagement', 'n_clicks')],
    [State('display-mode', 'data')]
)
def update_display_mode(raw_clicks, heatmap_clicks, engagement_clicks, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_mode
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return button_id.split('-')[-1]

@app.callback(
    Output('dummy-output', 'children'),
    Input('display-mode', 'data')
)
def debug_display_mode(display_mode):
    print(f"Display mode updated to: {display_mode}")
    return ""

@app.callback(
    [Output('video-display', 'children'),
     Output('engagement-chart', 'figure'),
     Output('attention-trend', 'figure'),
     Output('engagement-alert', 'children'),
     Output('historical-data', 'data'),
     Output('session-data', 'data'),
     Output('overall-attention-score', 'children'),
     Output('student-scores-chart', 'figure'),
     Output('student-selector', 'options'),
     Output('student-selector', 'value')],
    [Input('live-update', 'n_intervals'),
     Input('display-mode', 'data'),
     Input('frame-selector', 'value'),
     Input('upload-timestamp', 'data')],
    [State('historical-data', 'data'),
     State('session-data', 'data'),
     State('upload-data', 'data')]
)
def update_dashboard(n, display_mode, selected_frame, upload_timestamp, historical_data, session_data, upload_data):
    print(f"update_dashboard called with display_mode: {display_mode}, selected_frame: {selected_frame}")
    print(f"upload_data: {upload_data}")
    global attention_data
    
    current_upload_id = upload_data.get('upload_id')
    frame_count = upload_data.get('frame_count', 0)
    selected_frame = min(selected_frame, frame_count - 1) if frame_count > 0 else 0
    
    if not current_upload_id:
        print("No upload_id, returning default values")
        return (
            html.P("Upload an image or video to view analysis", style={'textAlign': 'center', 'padding': '2rem'}),
            px.pie(values=[0, 1], names=["Engaged", "Not Engaged"]).update_layout(margin=dict(l=20, r=20, t=50, b=20)),
            px.line().update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(range=[0, 1])),
            html.Div(),
            historical_data,
            session_data,
            "N/A",
            go.Figure(),
            [],
            None
        )
    
    # Use the cache by calling the function
    cache_key = f"{current_upload_id}_{selected_frame}"
    attention_data = analyze_frame_cache(cache_key)
    
    avg_score = 0.0
    scores = []
    names = []
    student_options = []
    default_value = None
    
    if attention_data:
        avg_score = attention_data.get('frame_stats', {}).get('mean_attention', 0.0)
        individual_scores = attention_data.get('individual_scores', [])
        scores = [score.get('attention_score', 0.0) for score in individual_scores]
        names = [f"Person ID: {score.get('person_idx', '')}" for score in individual_scores]
        student_options = [
            {'label': f"Person ID: {score.get('person_idx', '')}", 'value': score.get('person_idx', '')}
            for score in individual_scores
        ]
        default_value = student_options[0]['value'] if student_options else None
    else:
        print("No attention_data, using default values")
    
    historical_data['timestamps'].append(selected_frame)
    historical_data['scores'].append(avg_score)
    if len(historical_data['timestamps']) > 20:
        historical_data['timestamps'].pop(0)
        historical_data['scores'].pop(0)
    
    # Generate display based on mode
    try:
        if display_mode == 'raw':
            video_display = generate_live_view(current_upload_id, selected_frame)
        elif display_mode == 'heatmap':
            video_display = generate_heatmap_view(current_upload_id, selected_frame)
        else:  # engagement
            video_display = generate_engagement_view(current_upload_id, selected_frame)
    except Exception as e:
        print(f"Error generating video display ({display_mode}): {str(e)}")
        video_display = html.P(f"Error loading display: {str(e)}", style={'textAlign': 'center', 'padding': '2rem'})
    
    engagement_fig = generate_engagement_figure(avg_score)
    try:
        trend_fig = generate_trend_figure(current_upload_id, frame_count)
    except Exception as e:
        print(f"Error generating trend figure: {str(e)}")
        trend_fig = px.line().update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[0, 1]),
            annotations=[dict(text=f"Trend error: {str(e)}", x=0.5, y=0.5, showarrow=False)]
        )
    
    alert = generate_alert(avg_score)
    
    # Student scores chart
    student_fig = go.Figure()
    if scores and names:
        student_fig.add_trace(go.Bar(
            x=names,
            y=scores,
            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(names)],
            marker_line_width=0
        ))
    student_fig.update_layout(
        xaxis_title="Person",
        yaxis_title="Attention Score",
        yaxis_range=[0, 1],
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': '"Open Sans", sans-serif'}
    )
    
    return (
        video_display,
        engagement_fig,
        trend_fig,
        alert,
        historical_data,
        session_data,
        f"{avg_score:.0%}" if attention_data else "N/A",
        student_fig,
        student_options,
        default_value
    )

@app.callback(
    Output('student-details', 'children'),
    [Input('student-selector', 'value'),
     Input('frame-selector', 'value')],
    [State('upload-data', 'data')]
)
def update_student_details(person_id, selected_frame, upload_data):
    print(f"Selected person_id: {person_id}, type: {type(person_id)}")
    
    if not upload_data.get('upload_id') or person_id is None:
        return html.Div("No person data available")
    
    # Fetch analysis for the selected frame
    cache_key = f"{upload_data.get('upload_id')}_{selected_frame}"
    frame_data = analyze_frame_cache(cache_key)
    
    if not frame_data:
        return html.Div("No data for this frame")
    
    # Find the selected person, ensuring type match
    person = next((score for score in frame_data.get('individual_scores', []) if str(score['person_idx']) == str(person_id)), None)
    if not person:
        return html.Div(f"Person {person_id} not found")
    
    score = person.get('attention_score', 0.0)
    name = f"Person ID: {person['person_idx']}"
    looking_at = person.get('looking_at', 'Unknown')
    
    # Fetch temporal data for trend
    try:
        temporal_response = requests.post(
            API_ANALYZE_TEMPORAL,
            json={'upload_id': upload_data.get('upload_id')},
            verify=False
        )
        if temporal_response.status_code == 200:
            trend_data = temporal_response.json().get('trend', [])
            person_trend = []
            for frame_idx in range(len(trend_data)):
                cache_key = f"{upload_data.get('upload_id')}_{frame_idx}"
                frame_data = analyze_frame_cache(cache_key)
                if frame_data:
                    frame_scores = frame_data.get('individual_scores', [])
                    person_score = next((s.get('attention_score', 0.0) for s in frame_scores if str(s['person_idx']) == str(person_id)), 0.0)
                    person_trend.append(person_score)
                else:
                    person_trend.append(0.0)
        else:
            person_trend = [score]
    except Exception as e:
        print(f"Error fetching temporal data: {e}")
        person_trend = [score]
    
    # Person details layout
    return html.Div([
        html.Div([
            html.Img(
                src=f"https://ui-avatars.com/api/?name={name.replace(' ', '+')}&background=random&size=100",
                style={'width': '80px', 'height': '80px', 'borderRadius': '50%', 'objectFit': 'cover', 'marginBottom': '1rem', 'boxShadow': '0 4px 10px rgba(0,0,0,0.1)', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto'}
            ),
            html.H4(name, style={'textAlign': 'center', 'margin': '0 0 0.5rem 0', 'color': '#2c3e50'}),
            html.Div(f"Person Index: {person['person_idx']}", style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '1.5rem', 'fontSize': '0.9rem'})
        ], style={'textAlign': 'center'}),
        html.Div([
            html.Div("Current Attention", style={'fontSize': '0.9rem', 'color': '#7f8c8d', 'textAlign': 'center', 'marginBottom': '0.5rem'}),
            html.Div([
                html.Div(f"{score:.0%}", style={'fontSize': '2rem', 'fontWeight': '700', 'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div([
                    html.I(className="fas fa-arrow-up" if score > 0.6 else "fas fa-arrow-down", style={'marginRight': '5px', 'color': '#2ecc71' if score > 0.6 else '#e74c3c'}),
                    f"Looking at: {looking_at}"
                ], style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '0.8rem'})
            ], style={'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '1.5rem', 'borderRadius': '10px', 'marginBottom': '1.5rem', 'boxShadow': 'inset 0 4px 15px rgba(0,0,0,0.05)'})
        ]),
        dcc.Graph(
            figure={
                'data': [{'x': list(range(len(person_trend))), 'y': person_trend, 'type': 'line', 'marker': {'color': '#3498db'}}],
                'layout': {
                    'xaxis': {'title': 'Frame'},
                    'yaxis': {'title': 'Attention Score', 'range': [0, 1]},
                    'margin': {'l': 40, 'r': 40, 't': 30, 'b': 30},
                    'height': 200,
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)'
                }
            },
            config={'displayModeBar': False}
        )
    ])

if __name__ == '__main__':
    app.run(debug=True)