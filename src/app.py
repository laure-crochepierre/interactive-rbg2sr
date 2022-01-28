# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression
import gc
import os
import re
import json
import pickle
import psutil
import signal
import dropbox
from sympy.parsing.sympy_parser import (parse_expr, function_exponentiation, implicit_application,
                                        implicit_multiplication, standard_transformations)

transformations = standard_transformations + (function_exponentiation, implicit_application, implicit_multiplication)
import time
import subprocess

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash import dash_table
from dash import html
from dash import dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"

import plotly.graph_objs as go

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

metrics = {"MSE": mean_squared_error, "R2": r2_score,
           "Pearson": lambda y, yhat: pearsonr(y, yhat)[0],
           "Spearman": lambda y, yhat: spearmanr(y, yhat)[0],
           }

waiter = [dbc.Container([dbc.Row(dbc.Col(dbc.Spinner(show_initially=True, color="secondary"),
                                         width={"size": 6, "offset": 6})),
                         dbc.Row(dbc.Col(html.H4("Waiting to receive expressions to compare"),
                                         width={"size": 6, "offset": 4}))],
                        className="p-3 bg-light rounded-3")]

# Launch app
app_name = __name__
if app_name == "__main__":
    app_name = "interactive_preference_app"
app = dash.Dash(app_name, external_stylesheets=[dbc.themes.LUMEN, FONT_AWESOME])
server = app.server

app.layout = html.Div([
    dbc.Row([dbc.Col(dbc.Button("Visualize expressions",
                                id="off-button",
                                color="primary", style={'margin': 30})),
             dbc.Col([
                 html.H1("Reinforcement Based Grammar Guided Symbolic Regression with Preference Learning",
                         style={'margin': 25, 'textAlign': 'center'}),
             ], width={'size': 8}),
             dbc.Col([dbc.Button("New training",  # 'Nouvel entrainement',
                                 id="new_training",
                                 color="primary", style={'margin': 30})
                      ])
             ]),
    dcc.Store(id='local-gui-data-logdir', storage_type='local'),
    dcc.Store(id='local-current-step', storage_type='local', data=0),
    dcc.Store(id='local-pid', storage_type='local'),
    dcc.Store(id='local-pair-indexes', storage_type='memory'),
    dcc.Store(id='local-grammar', storage_type='local'),
    html.Div([dbc.Row(dbc.Col([
        dbc.Form([
            dbc.Label('Dataset', html_for="dataset-input"),
            dcc.Dropdown(options=[{"label": 'Symbolic Regression benchmark', 'value': "nguyen4"},
                                  {"label": 'Power System Use case', 'value': "case14", "disabled": True}],
                         value="nguyen4",
                         id="dataset-input"),
            html.Br(),
            dbc.Label('Grammar', html_for="grammar-input"),
            dcc.Dropdown(options=[{"label": 'Grammar with constants', 'value': "with"},
                                  {"label": 'Grammar without constants', 'value': "without"}],
                         value="without",
                         id="grammar-input"),
            html.Br(),
            dbc.Label('Interaction Type', html_for="type-input", style="display: none"),
            dcc.Dropdown(options=[{"label": "From the start", "value": "from_start"},
                                  {"label": "On plateau", "value": "on_plateau"}],
                         value="from_start", id="type-input", style="display: none",
                         disabled=True),
            html.Br(style="display: none"),
            dbc.Label('Reuse Preferences between interactive iterations ? ', html_for="reuse-input"),
            dcc.Dropdown(options=[{"label": "Yes", "value": "yes"},
                                  {"label": "No", "value": "no"}],
                         value="yes", id="reuse-input"),
            html.Br(),
            dbc.Label('Interaction Frequency', html_for="frequency-input"),
            dcc.Slider(min=1, max=50, step=1, value=5, tooltip={"placement": "bottom", "always_visible": True},
                       id="frequency-input")
        ], style={"padding": '2%'}),
        dbc.Button("Start Training",  # "Commencer l'entrainement",
                   id="launch-training", n_clicks=0,
                   className="d-grid col-12 mx-auto",
                   color="primary", size="lg")],
        width={"size": 6, "offset": 3}))
    ], className="gap-2", id="before-training", hidden=True),
    html.Div([dcc.Interval(id="interval-during-training", interval=5 * 1000, disabled=True),
              html.Div(waiter, id='waiter'),
              html.Div([
                  dbc.Offcanvas(
                      [
                          dbc.Label("Select an expression", html_for="visualize-expression-dropdown"),
                          dcc.Dropdown(clearable=False,
                                       searchable=True,
                                       id="visualize-expression-dropdown"),
                          html.Br(),
                          dcc.Graph(id="visualize-expression-graph")
                      ],
                      id="offcanvas",
                      title="Visualize Expressions",
                      is_open=False,
                      close_button=False,
                      style={'width': '800px'}
                  ),
                  dcc.Tabs([
                      dcc.Tab(label="Categorical preferences",  # 'Préférences Hautes-Moyennes-Basses',
                              children=html.Div(id='preference-classes', className="h5"), className="h4"),
                      dcc.Tab(label="Preference pairs",  # 'Paires de préférences',
                              children=
                              dbc.Row([
                                  dbc.Col(
                                      dash_table.DataTable(
                                          columns=([{'id': 'id', 'name': 'id'},
                                                    {'id': 'Expression', 'name': 'Expression'},
                                                    {'id': 'Reward', 'name': 'Reward'}]),
                                          data=[],
                                          editable=False,
                                          row_selectable="multi",
                                          sort_action='native',
                                          filter_action='native',
                                          style_cell={'whiteSpace': 'pre-line'},
                                          id="datatable"
                                      ), width={"size": 4}
                                  ),
                                  dbc.Col([
                                      dbc.Row(dbc.Col(html.Div(id='all-preference-pairs')))
                                  ], width={"size": 8})
                              ], style={"padding-top": "2%"}, className="h5"),
                              className="h4"),
                      dcc.Tab(label="Solution suggestion",  # "Suggestion d'une solution",
                              children=html.Div(id='solution-suggestion', className="h5"), className="h4")
                  ]),
                  dbc.Row(dbc.Col(html.Div(id='valider-et-continuer'), width={"size": 8, "offset": 2})),
              ], id="iteration_data")],
             hidden=True, id="during-training", style={'margin': 25})
])


@app.callback([Output('offcanvas', "is_open")],
              Input('off-button', "n_clicks"))
def open_off_canevas(n_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == ".":
        raise PreventUpdate
    return [True]


@app.callback(Output("off-button", "style"),
              [Input("iteration_data", "hidden"),
               Input("before-training", "hidden")])
def is_canevas_hidden(hidden_iteration, hidden_before):
    if (not hidden_before) or hidden_iteration:
        return {"display": "none"}
    return {'margin': 30}


@app.callback([Output('interval-during-training', 'disabled'),
               Output("during-training", "hidden"),
               Output("before-training", "hidden"),
               Output("waiter", "hidden"),
               Output("iteration_data", "hidden"),
               Output('local-gui-data-logdir', "data"),
               Output('local-pid', "data"),
               Output('local-pair-indexes', "data"),
               Output('local-current-step', "data"),
               Output("local-grammar", "data"),
               Output("datatable", "data"),
               Output("all-preference-pairs", "children"),
               Output("solution-suggestion", "children"),
               Output("preference-classes", "children"),
               Output("valider-et-continuer", "children"),
               Output("datatable", "selected_row_ids"),
               Output('visualize-expression-dropdown', 'options'),
               Output('visualize-expression-graph', 'figure')
               ],
              [Input('launch-training', 'n_clicks'),
               Input('interval-during-training', 'n_intervals'),
               Input({'type': 'validate', 'index': ALL}, 'n_clicks'),
               Input({'type': 'delete_pair', 'index': ALL}, 'n_clicks'),
               Input("datatable", "selected_row_ids"),
               Input('new_training', "n_clicks"),
               Input('visualize-expression-dropdown', "value"),
               Input('type-input', "value"),
               Input('reuse-input', "value"),
               ],
              [State('local-gui-data-logdir', "data"),
               State('local-current-step', "data"),
               State('local-pid', "data"),
               State('local-pair-indexes', "data"),
               State("local-grammar", "data"),
               State({'type': "local-expression-data", 'index': ALL}, "data"),
               State("datatable", "data"),
               State("all-preference-pairs", "children"),
               State("solution-suggestion", "children"),
               State("preference-classes", "children"),
               State("valider-et-continuer", "children"),
               State({'type': 'prefer_right', 'index': ALL}, 'n_clicks'),
               State({'type': 'prefer_left', 'index': ALL}, 'n_clicks'),
               State({'type': 'prefer_both', 'index': ALL}, 'n_clicks'),
               State({'type': 'prefer_none', 'index': ALL}, 'n_clicks'),
               State({'type': 'top-expression-ids', 'index': ALL}, 'data'),
               State({'type': 'middle-expression-ids', 'index': ALL}, 'data'),
               State({'type': 'low-expression-ids', 'index': ALL}, 'data'),
               State('dataset-input', "value"),
               State('grammar-input', "value"),
               State('frequency-input', "value")
               ]
              )
def content_callback(launch_n_clicks, n_intervals, validate_n_clicks, delete_pair_n_clicks, selected_row_indices,
                     new_clicks, visu_dropdown_value, interaction_type, reuse,
                     logdir, current_step, pid, pair_indexes, grammar, local_expression_data, table_data,
                     children, suggestion_box, pref_classes, continuer_box, right_pref, left_pref, both_pref, none_pref,
                     top_idss, middle_idss, low_idss, dataset_value, grammar_value, frequency_value):
    hidden_during = dash.no_update
    hidden_before = dash.no_update
    hidden_waiter = dash.no_update
    hidden_iteration_data = dash.no_update
    interval_disabled = dash.no_update
    visu_dropdown = dash.no_update
    visu_graph = dash.no_update

    top_ids = []
    middle_ids = []
    low_ids = []
    if (top_idss is not None) and (len(top_idss) > 0):
        top_ids = top_idss[-1]
    if (middle_idss is not None) and (len(middle_idss) > 0):
        middle_ids = middle_idss[-1]
    if (low_idss is not None) and (len(low_idss) > 0):
        low_ids = low_idss[-1]

    if (pid is not None) and (pid > 0) and not psutil.pid_exists(pid):
        try:
            pid = -1
            hidden_before = False
            hidden_waiter = True
            hidden_during = True
            hidden_iteration_data = True
            interval_disabled = True
            pair_indexes = []
            selected_row_indices = []
            grammar = None
            logdir = None
        except:
            print('Not existing')

        return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
               pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
               continuer_box, selected_row_indices, visu_dropdown, visu_graph

    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == "new_training.n_clicks":
        if (pid is not None) and (pid > 0):
            proc = psutil.Process(pid)
            children_proc = proc.children(recursive=True) + [proc]
            for p in children_proc:
                try:
                    p.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
                gone, alive = psutil.wait_procs(children_proc)
        pid = -1
        hidden_before = False
        hidden_waiter = False
        hidden_during = True
        hidden_iteration_data = True
        interval_disabled = True
        pair_indexes = []
        selected_row_indices = []
        grammar = None
        logdir = None

    elif ctx.triggered[0]['prop_id'] == "launch-training.n_clicks":
        interval_disabled, hidden_during, hidden_before, logdir, pid = callback_launch(dataset_value,
                                                                                       grammar_value,
                                                                                       frequency_value,
                                                                                       interaction_type, reuse)
        current_step = 0
        hidden_iteration_data = True
    elif "validate" in ctx.triggered[0]['prop_id']:
        validate_prefs(logdir, pair_indexes, current_step, right_pref, left_pref, both_pref, none_pref,
                       top_ids, middle_ids, low_ids, local_expression_data)
        interval_disabled = False
        hidden_iteration_data = True
        hidden_during = False
        children = []
        pair_indexes = None
    elif "delete_pair" in ctx.triggered[0]['prop_id']:
        pair_id_to_drop = json.loads(ctx.triggered[0]['prop_id'].replace('.n_clicks', ''))['index']
        pair_indexes.pop([str(p) for p in pair_indexes].index(pair_id_to_drop))
        children, continuer_box, grammar, table_data, pair_indexes, \
        interval_disabled, hidden_iteration_data, hidden_during, current_step, visu_dropdown, visu_graph = pairs_plot_callback(
            n_intervals, logdir, current_step, pair_indexes, grammar, visu_dropdown_value)

    elif "datatable.selected_row_ids" in ctx.triggered[0]['prop_id']:
        new_pair = table_callback(selected_row_indices)
        if new_pair is not None:
            selected_row_indices = []
            pair_indexes = new_pair + pair_indexes
            children, continuer_box, grammar, table_data, pair_indexes, \
            interval_disabled, hidden_iteration_data, hidden_during, current_step, visu_dropdown, visu_graph = pairs_plot_callback(
                n_intervals, logdir, current_step, pair_indexes, grammar, visu_dropdown_value)
    elif (pid is not None) and (pid > 0) and psutil.pid_exists(pid):
        if (not pair_indexes is None) and (len(pair_indexes) == 0):
            pair_indexes = None
        children, continuer_box, grammar, table_data, pair_indexes, \
        interval_disabled, hidden_iteration_data, hidden_during, current_step, visu_dropdown, visu_graph = pairs_plot_callback(
            n_intervals, logdir, current_step, pair_indexes, grammar, visu_dropdown_value)
        pref_classes = get_classes(table_data, current_step, top_ids, middle_ids, low_ids)

        if not isinstance(grammar, dict):
            return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
                   pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
                   continuer_box, selected_row_indices, visu_dropdown, visu_graph
        expression = grammar['start_symbol']
        current_symbol = grammar['start_symbol']
        queue = []
        action_ids = []
        if (local_expression_data is None) or (local_expression_data == []):
            local_expression_data = [{"translation": expression, "current_symbol": current_symbol, "queue": queue,
                                      "action_ids": action_ids}]
        else:

            local_expression_data = [local_expression_data[-1]]
            expression = local_expression_data[-1]['translation']
            current_symbol = local_expression_data[-1]['current_symbol']

        grammar_content = []
        for key, prod in grammar['productions_dict'].items():
            grammar_content += [html.I(f"{key} :== {' | '.join([pr['raw'] for pr in prod])}")] + [html.Br()]

        grammar_div = dbc.Alert([html.P('Grammar reminder')] + grammar_content, color="secondary")

        suggestion_box = dbc.Row(
            [dbc.Col(
                dbc.Alert([
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                html.Div([html.P("Choose an expression to compare with"),
                                          # "Choisir une expression avec laquelle se comparer"),
                                          dcc.Dropdown(id={'type': "suggestion-id", 'index': current_step},
                                                       options=[
                                                           {"label": f"{row['Expression']} (score {row['Reward']})",
                                                            "value": row['id']} for row in table_data]),
                                          html.P(" ")], id={'type': "validation-div", 'index': current_step})
                            )
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div([html.P("Suggest an expression"),  # "Suggestion d'une expression"),
                                      dbc.Alert("Start expression: " + expression, color="light",
                                                style={'color': "black",
                                                       "background-color": "white",
                                                       "border-width": "0px"},
                                                id={'type': "expression", 'index': current_step})],
                                     id={'type': "alert-div", 'index': current_step})
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div([
                                dcc.Store(id={'type': "local-expression-data", 'index': current_step},
                                          storage_type='memory',
                                          data=local_expression_data[-1]),
                                html.Div(
                                    dcc.Dropdown(
                                        id={'type': "select-action", 'index': current_step},
                                        options=[{"label": rule['raw'], "value": i_prod}
                                                 for i_prod, rule in enumerate(grammar['productions_list'])
                                                 if current_symbol in rule["parent_symbol"]
                                                 ]),
                                    id={'type': "dropdown-div", 'index': current_step})
                            ])
                        )
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Start a new suggestion",  # "Commencer une nouvelle suggestion",
                                       id={'type': 'restart-suggest',
                                           'index': current_step},
                                       className="d-grid gap-2 col-6 mx-auto",
                                       style={"margin": "1%"})
                        ),
                        dbc.Col(
                            dbc.Button("Suggestion validation",  # "Valider la suggestion",
                                       id={'type': 'suggestion-validation',
                                           'index': current_step},
                                       color="primary",
                                       className="d-grid gap-2 col-6 mx-auto",
                                       style={"margin": "1%"}, disabled=True)),
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div(id={"type": "validated-suggestion-div", "index": current_step})])
                    ])
                ], color="secondary")
            ),
                dbc.Col(grammar_div)]
        )

        if isinstance(children, list) and (len(children) > 0):
            hidden_during = False
            hidden_iteration_data = False
            hidden_waiter = True
    elif pid is None:
        hidden_before = False
        return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
               pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
               continuer_box, selected_row_indices, visu_dropdown, visu_graph

    hidden_before = not hidden_during
    hidden_waiter = not hidden_iteration_data
    gc.collect()
    return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
           pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
           continuer_box, selected_row_indices, visu_dropdown, visu_graph


def callback_launch(dataset_value, grammar_value, frequency_value, interaction_type, reuse):

    if os.environ.get('DROPBOX_ACCESS_TOKEN') is not None:
        writer_logdir = f"/results/interactive_runs/{time.time()}"
        gui_data_logdir = os.path.join(writer_logdir, 'gui_data')
        dbx = dropbox.Dropbox(os.environ.get('DROPBOX_ACCESS_TOKEN'))
        py_script = 'src/app_interactive_algorithm.py'
    else:
        writer_logdir = f"../results/interactive_runs/{time.time()}"
        gui_data_logdir = os.path.join(writer_logdir, 'gui_data')
        os.makedirs("../results/interactive_runs", exist_ok=True)
        os.makedirs(gui_data_logdir, exist_ok=True)
        py_script = 'app_interactive_algorithm.py'
    proc = subprocess.Popen([f'python {py_script} {writer_logdir} {dataset_value} '
                             f'{grammar_value} {frequency_value} {interaction_type} {reuse}'], shell=True)
    print("Training Launched !")
    return False, False, True, gui_data_logdir, proc.pid


def expression_formating(t):
    if (t == '') or ('<' in t):
        return ''

    t = t.replace('np.', '')
    for i in range(10):
        t = t.replace(f'x[:,{i}]', f"x{i}").replace(f'x[:, {i}]', f"x{i}")
    t = parse_expr(t).__repr__()
    return t


def pairs_plot_callback(n_intervals, gui_data_logdir, current_step, combinaisons=None, grammar=None,
                        visu_dropdown_value=None):
    """

    :param n_intervals:
    :param gui_data_logdir:
    :param current_step:
    :param combinaisons:
    :param grammar:
    :param visu_dropdown_value:
    :return:
    - children
    - continuer_box
    - grammar
    - table_data
    - pair_indexes
    - interval_disabled
    - hidden_iteration_data
    - hidden_during
    - current_step
    - visu_dropdown
    - visu_graph

    """
    gui_data_logdir_list = []
    if os.environ.get("DROPBOX_ACCESS_TOKEN") is None:
        gui_data_logdir_list = os.listdir(gui_data_logdir)
    else:
        dbx = dropbox.Dropbox(os.environ.get("DROPBOX_ACCESS_TOKEN"))
        try:
            gui_data_logdir_list = [e.name for e in dbx.files_list_folder(gui_data_logdir).entries]
        except:
            raise PreventUpdate

    if len(gui_data_logdir_list) == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    input_pkls = [os.path.join(gui_data_logdir, f) for f in gui_data_logdir_list
                  if not "answers" in f]
    answers_pkls = [os.path.join(gui_data_logdir, f) for f in gui_data_logdir_list
                    if "answers" in f]

    if len(input_pkls) - len(answers_pkls) == 0:
        return [], [], grammar, [], [], False, True, False, current_step, dash.no_update, dash.no_update
    else:
        current_step = max([int(f.split('/')[-1].replace(".pkl", "")) for f in input_pkls])

    pairs_data_path = os.path.join(gui_data_logdir, f"{current_step}.pkl")

    pairs_data = None
    if os.environ.get("DROPBOX_ACCESS_TOKEN") is None:
        pairs_data = pickle.load(open(pairs_data_path, 'rb'))
    else:
            _, file_content = dbx.files_download(pairs_data_path)
            pairs_data = pickle.loads(file_content.content)

    if combinaisons is None:
        combinaisons = [list(p) for p in pairs_data['combinaisons']]
    rewards = pairs_data['rewards']
    translations = pairs_data['translations']
    y_pred = pairs_data['predicted_values']
    x = pairs_data['x']
    y = pairs_data['target_values']
    top_indices = pairs_data['top_indices'].T
    if grammar is None:
        grammar = pairs_data['grammar']

    translations = [t.replace(']]', ']').replace('x.columns[', ':,') for t in translations]
    translations = [expression_formating(t) for t in translations]
    top_expressions = sorted(list(set([translations[i] for i in top_indices])),
                             key=lambda t: rewards[translations.index(t)], reverse=True)

    table_data = [{'id': translations.index(t),
                   'Expression': t,
                   'Reward': round(rewards[translations.index(t)], 2)}
                  for t in top_expressions]

    if visu_dropdown_value is None:
        visu_dropdown_value = translations.index(top_expressions[0])

    visu_options = [{"label": f"{t} (score {round(rewards[translations.index(t)], 3)})",
                     "value": translations.index(t)}
                    for t in top_expressions]
    visu_figure = go.Figure(data=[go.Scatter(x=x, y=y_pred[visu_dropdown_value],
                                             name="y_pred", mode='markers'),
                                  go.Scatter(x=x, y=y, name="y", mode='markers')],
                            layout=go.Layout(xaxis_title="Variable x",
                                             title=translations[visu_dropdown_value],
                                             autosize=True,
                                             margin=go.layout.Margin(l=1, r=1, b=1, t=50)))

    children_blocks = [html.H4(f"Iteration n°{current_step}", style={"textAlign": 'center'})]
    for i_pair, pair_ids in enumerate(combinaisons):
        id_left, id_right = pair_ids
        pair_ids = str(pair_ids)
        r_left, r_right = round(rewards[id_left], 3), round(rewards[id_right], 3)
        t_left = translations[id_left]
        t_right = translations[id_right]

        left_fig = go.Figure(data=[go.Scatter(x=x, y=y_pred[id_left], name="y_pred", mode='markers'),
                                   go.Scatter(x=x, y=y, name="y", mode='markers')],
                             layout=go.Layout(xaxis_title="Variable x",
                                              margin=go.layout.Margin(l=1, r=1, b=1, t=1)))

        left_fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        toast_content_left = [
            dbc.Row(dbc.Col(dcc.Graph(figure=left_fig)
                            )
                    ),
            # dbc.Row(dbc.Col([f"Score de récompense : ", html.Strong(r_left)])),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id=f'table-results-left-{i_pair}',
                columns=([{'id': 'Reward', 'name': 'Reward'}] + [{'id': m, 'name': m} for m in metrics.keys()]),
                data=[dict(Reward=r_left, **{name: round(m(y, y_pred[id_left]), 2) for name, m in metrics.items()})],
                editable=False
            ))),
            dbc.Row(dbc.Col(dbc.Button("Choose",  # "Choisir",
                                       color="secondary", className="d-grid col-12 mx-auto",
                                       id={"type": 'prefer_left', "index": pair_ids}), width={"size": 3, "offset": 5}))]

        right_fig = go.Figure(data=[go.Scatter(x=x, y=y_pred[id_right], name="y_pred", mode='markers'),
                                    go.Scatter(x=x, y=y, name="y", mode='markers')],
                              layout=go.Layout(xaxis_title="Variable x",
                                               margin=go.layout.Margin(l=0, r=0, b=0, t=0)))
        right_fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        toast_content_right = [
            dbc.Row(dbc.Col(dcc.Graph(figure=right_fig))),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id='table-editing-simple',
                columns=([{'id': 'Reward', 'name': 'Reward'}] + [{'id': m, 'name': m} for m in metrics.keys()]),
                data=[dict(Reward=r_right, **{name: round(m(y, y_pred[id_right]), 2) for name, m in metrics.items()})],
                editable=False
            ))),
            dbc.Row(dbc.Col(dbc.Button("Choose",  # "Choisir",
                                       color="secondary",
                                       className="d-grid col-12 mx-auto",
                                       id={"type": 'prefer_right', "index": pair_ids}),
                            width={"size": 3, "offset": 4}))]
        left_toast = dbc.Toast(toast_content_left,
                               header=t_left,
                               style={"width": "100%", "max-width": "48%", "margin": "1%", "font-size": "1rem"},
                               header_class_name="h5")
        right_toast = dbc.Toast(toast_content_right,
                                header=t_right,
                                style={"width": "100%", "max-width": "48%", "margin": "1%", "font-size": "1rem"},
                                header_class_name="h5")
        pair_row = dbc.Form([dbc.Row([left_toast, right_toast]),
                             dbc.Row(dbc.Col(dbc.Button("Equivalent",
                                                        color="secondary",
                                                        className="d-grid gap-2 col-6 mx-auto",
                                                        style={"margin": "1%"},
                                                        id={"type": 'prefer_both', "index": pair_ids}),
                                             width={"size": 8, "offset": 2})),
                             dbc.Row(dbc.Col(dbc.Button("Neither",  # "Je n'aime aucun des 2",
                                                        color="success",
                                                        className="d-grid gap-2 col-6 mx-auto",
                                                        style={"margin": "1%"},
                                                        id={"type": 'prefer_none', "index": pair_ids}),
                                             width={"size": 8, "offset": 2}))
                             ])
        card_header = dbc.Row([dbc.Col([html.H4(f'Pair {pair_ids}')], width={"size": 10}),
                               dbc.Col([dbc.Button(html.I(className="far fa-times-circle fa-lg",
                                                          **{'aria-hidden': 'true'},
                                                          children=None),
                                                   id={'type': 'delete_pair', 'index': pair_ids})])])
        children_blocks.append(dbc.Card([dbc.CardHeader([card_header],
                                                        style={"text-align": "center"}),
                                         dbc.CardBody(pair_row)]))
    continuer_box = dbc.Button("Validate and continue",  # "Valider et continuer",
                               id={'type': 'validate',
                                   'index': current_step},
                               color="dark",
                               className="d-grid gap-2 col-6 mx-auto",
                               style={"margin": "1%"})

    return children_blocks, continuer_box, grammar, table_data, combinaisons, True, False, False, current_step, \
           visu_options, visu_figure


@app.callback([Output({'type': "dropdown-div", 'index': MATCH}, "children"),
               Output({'type': "local-expression-data", 'index': MATCH}, "data"),
               Output({'type': "expression", 'index': MATCH}, "children"),
               Output({'type': "suggestion-validation", 'index': MATCH}, "disabled"),
               Output({'type': "validated-suggestion-div", 'index': MATCH}, "children")
               ],
              [Input({'type': "select-action", 'index': MATCH}, "value"),
               Input({'type': "restart-suggest", 'index': MATCH}, "n_clicks"),
               Input({'type': 'suggestion-validation', 'index': MATCH}, "n_clicks"),
               Input({'type': "suggestion-id", 'index': MATCH}, "value")],
              [State({'type': "local-expression-data", 'index': MATCH}, "data"),
               State("local-grammar", "data"),
               State('local-current-step', "data"),
               ])
def update_selected_action(selected_value, n_clicks_restart, n_clicks_suggestion_validation, suggestion_id,
                           local_expression_data, grammar, current_step):
    disabled_validation = ('<' in local_expression_data['translation']) or (suggestion_id is None)

    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == ".":
        if grammar is None:
            raise PreventUpdate
        local_expression_data['queue'] = []
        local_expression_data['translation'] = grammar['start_symbol']
        local_expression_data['current_symbol'] = grammar['start_symbol']
        local_expression_data['action_ids'] = []
        local_expression_data['comparison_with_id'] = None

        gc.collect()
        return dash.no_update, local_expression_data, dash.no_update, disabled_validation, dash.no_update
    elif "restart-suggest" in ctx.triggered[0]['prop_id']:
        local_expression_data['queue'] = []
        local_expression_data['translation'] = grammar['start_symbol']
        local_expression_data['current_symbol'] = grammar['start_symbol']
        local_expression_data['action_ids'] = []
        local_expression_data['comparison_with_id'] = suggestion_id

        options = [{"label": rule['raw'], "value": i_prod}
                   for i_prod, rule in enumerate(grammar['productions_list'])
                   if grammar['start_symbol'] in rule["parent_symbol"]]

        expression = local_expression_data['translation']
        dd = dcc.Dropdown(
            id={'type': "select-action", 'index': current_step},
            options=options)

        gc.collect()
        return dd, local_expression_data, expression, True, []
    elif "suggestion-validation" in ctx.triggered[0]['prop_id']:

        gc.collect()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               [str(n_clicks_suggestion_validation) + str(local_expression_data)]
    elif "suggestion-id" in ctx.triggered[0]['prop_id']:
        local_expression_data['comparison_with_id'] = suggestion_id

        gc.collect()
        return dash.no_update, local_expression_data, dash.no_update, disabled_validation, dash.no_update

    if selected_value is not None:
        rule = grammar['productions_list'][selected_value]

        local_expression_data['translation'] = local_expression_data['translation'].replace(
            local_expression_data['current_symbol'], rule['raw'], 1)
        local_expression_data['action_ids'].append(selected_value)

        local_expression_data['queue'] = rule['descendant_symbols'] + local_expression_data['queue']

    if len(local_expression_data['queue']) == 0:
        expression = local_expression_data['translation']
        disabled_validation = ('<' in local_expression_data['translation']) or (suggestion_id is None)

        gc.collect()
        return dash.no_update, local_expression_data, expression, disabled_validation, dash.no_update
    else:
        next_symbol = local_expression_data['queue'].pop(0)

        local_expression_data['current_symbol'] = next_symbol
        options = [{"label": r['raw'], "value": i_prod}
                   for i_prod, r in enumerate(grammar['productions_list'])
                   if next_symbol in r["parent_symbol"]]

        expression = local_expression_data['translation']
        dd = dcc.Dropdown(id={'type': "select-action", 'index': current_step}, options=options),
        disabled_validation = ('<' in local_expression_data['translation']) or (suggestion_id is None)

        gc.collect()
        return dd, local_expression_data, expression, disabled_validation, dash.no_update


@app.callback(
    [Output({'type': 'prefer_right', 'index': MATCH}, 'color'),
     Output({'type': 'prefer_left', 'index': MATCH}, 'color'),
     Output({'type': 'prefer_both', 'index': MATCH}, 'color'),
     Output({'type': 'prefer_none', 'index': MATCH}, 'color'),
     Output({'type': 'prefer_right', 'index': MATCH}, 'n_clicks'),
     Output({'type': 'prefer_left', 'index': MATCH}, 'n_clicks'),
     Output({'type': 'prefer_both', 'index': MATCH}, 'n_clicks'),
     Output({'type': 'prefer_none', 'index': MATCH}, 'n_clicks'),
     ],
    [Input({'type': 'prefer_right', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'prefer_left', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'prefer_both', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'prefer_none', 'index': MATCH}, 'n_clicks'),
     ]
)
def color_preference_right_callback(n_right, n_left, n_both, n_none):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    colors = ["secondary", "secondary", "secondary", "secondary"]
    select_color = "success"
    n_clicks = [None, None, None, None]
    if "right" in ctx.triggered[0]['prop_id']:
        colors[0] = select_color
        n_clicks[0] = 1
    elif "left" in ctx.triggered[0]['prop_id']:
        colors[1] = select_color
        n_clicks[1] = 1
    elif "both" in ctx.triggered[0]['prop_id']:
        colors[2] = select_color
        n_clicks[2] = 1
    elif "none" in ctx.triggered[0]['prop_id']:
        colors[3] = select_color
        n_clicks[3] = 1
    return colors + n_clicks


def validate_prefs(logdir, pair_indexes, current_step, right_pref, left_pref, both_pref, none_pref,
                   top_ids, middle_ids, low_ids, suggestions):
    right_pref = [{True: '', False: 'r'}[answer is None] for answer in right_pref]
    left_pref = [{True: '', False: 'l'}[answer is None] for answer in left_pref]
    both_pref = [{True: '', False: 'b'}[answer is None] for answer in both_pref]
    none_pref = [{True: '', False: 'n'}[answer is None] for answer in none_pref]
    prefs = {"pairs": {"ids": pair_indexes,
                       "answers": [r + l + b + n for r, l, b, n in zip(right_pref, left_pref, both_pref, none_pref)]},
             "classes": {"top": top_ids, "middle": middle_ids, "low": low_ids},
             "suggest": [s for s in suggestions if ("<" not in s['translation']) and
                         (s['comparison_with_id'] is not None)]}
    answers_path = os.path.join(logdir, f"{current_step}_answers.pkl")
    if os.environ.get('DROPBOX_ACCESS_TOKEN') is None:
        pickle.dump(prefs, open(answers_path, 'wb'))
    else:
        dbx = dropbox.Dropbox(os.environ.get('DROPBOX_ACCESS_TOKEN'))
        dbx.files_upload(pickle.dumps(prefs), path=answers_path)

    gc.collect()


def table_callback(selected_row_indices):
    if len(selected_row_indices) == 2:
        return [selected_row_indices]
    else:
        raise PreventUpdate


def get_classes(table_data, current_step, top_ids, middle_ids, low_ids):
    if not isinstance(table_data, list):
        return dash.no_update

    filter_options = [{"label": "Contains", "value": "(STR)"},
                      {"label": "Does not contains", "value": "^((?!(STR)).)*$"},
                      {"label": "All remaining", "value": ".*"}]
    classes_ranking = [dbc.Col([
        dbc.Row([dbc.Col(html.H4("Best expressions")),  # "Meilleures expressions")),
                 dbc.Col(html.H4("Average expressions")),  # "Expressions moyennes")),
                 dbc.Col(html.H4("Bad expressions"))]),  # "Expression mauvaises"))]),
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.P('Selection by filter (top solutions)'),
                    dcc.Dropdown(placeholder="Select filter type",
                                 id={'type': "top-regex-type", 'index': current_step},
                                 options=filter_options),
                    dbc.Input(id={'type': "top-regex-input", 'index': current_step},
                              placeholder="String to select",
                              style={'margin-top': '2%', "margin-bottom": '2%'}),

                    dbc.Button("Apply selection", id={'type': "top-regex-apply-button", 'index': current_step}),
                ], color="secondary"),
                dcc.Dropdown(id={'type': "top-class", 'index': current_step},
                             multi=True,
                             searchable=True,
                             clearable=False,
                             options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                       "value": row['id']} for row in table_data],
                             value=top_ids)
            ]),
            dbc.Col([
                dbc.Alert([
                    html.P('Selection by filter (middle solutions)'),
                    dcc.Dropdown(placeholder="Select filter type",
                                 id={'type': "middle-regex-type", 'index': current_step},
                                 options=filter_options),
                    dbc.Input(id={'type': "middle-regex-input", 'index': current_step},
                              placeholder="String to select",
                              style={'margin-top': '2%', "margin-bottom": '2%'}),

                    dbc.Button("Apply selection", id={'type': "middle-regex-apply-button", 'index': current_step}),
                ], color="secondary"),
                dcc.Dropdown(id={'type': "middle-class", 'index': current_step},
                             multi=True,
                             searchable=True,
                             clearable=False,
                             options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                       "value": row['id']} for row in table_data],
                             value=middle_ids)
            ]),
            dbc.Col([
                dbc.Alert([
                    html.P('Selection by filter (low solutions)'),
                    dcc.Dropdown(placeholder="Select filter type",
                                 id={'type': "low-regex-type", 'index': current_step},
                                 options=filter_options),
                    dbc.Input(id={'type': "low-regex-input", 'index': current_step},
                              placeholder="String to select",
                              style={'margin-top': '2%', "margin-bottom": '2%'}),

                    dbc.Button("Apply selection", id={'type': "low-regex-apply-button", 'index': current_step}),
                ], color="secondary"),
                dcc.Dropdown(id={'type': "low-class", 'index': current_step},
                             multi=True,
                             searchable=True,
                             clearable=False,
                             options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                       "value": row['id']
                                       } for row in table_data],
                             value=low_ids)
            ])
        ]),

    ],
        style={'padding': '1%'}
    ),
        dcc.Store(id={'type': 'top-expression-ids', 'index': current_step}, storage_type="memory", data=[]),
        dcc.Store(id={'type': 'middle-expression-ids', 'index': current_step}, storage_type="memory", data=[]),
        dcc.Store(id={'type': 'low-expression-ids', 'index': current_step}, storage_type="memory", data=[])]
    return classes_ranking


@app.callback(
    [Output({'type': 'top-class', 'index': MATCH}, 'options'),
     Output({'type': 'middle-class', 'index': MATCH}, 'options'),
     Output({'type': 'low-class', 'index': MATCH}, 'options'),
     Output({'type': 'top-class', 'index': MATCH}, 'value'),
     Output({'type': 'middle-class', 'index': MATCH}, 'value'),
     Output({'type': 'low-class', 'index': MATCH}, 'value'),
     Output({'type': 'top-expression-ids', 'index': MATCH}, 'data'),
     Output({'type': 'middle-expression-ids', 'index': MATCH}, 'data'),
     Output({'type': 'low-expression-ids', 'index': MATCH}, 'data'),
     ],
    [Input({'type': 'top-class', 'index': MATCH}, 'value'),
     Input({'type': 'middle-class', 'index': MATCH}, 'value'),
     Input({'type': 'low-class', 'index': MATCH}, 'value'),
     Input({'type': 'top-regex-apply-button', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'middle-regex-apply-button', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'low-regex-apply-button', 'index': MATCH}, 'n_clicks'),
     ],
    [State({'type': 'top-expression-ids', 'index': MATCH}, 'data'),
     State({'type': 'middle-expression-ids', 'index': MATCH}, 'data'),
     State({'type': 'low-expression-ids', 'index': MATCH}, 'data'),
     State("datatable", "data"),
     State({'type': 'top-regex-type', 'index': MATCH}, 'value'),
     State({'type': 'top-regex-input', 'index': MATCH}, 'value'),
     State({'type': 'middle-regex-type', 'index': MATCH}, 'value'),
     State({'type': 'middle-regex-input', 'index': MATCH}, 'value'),
     State({'type': 'low-regex-type', 'index': MATCH}, 'value'),
     State({'type': 'low-regex-input', 'index': MATCH}, 'value'),
     ]
)
def pairs_by_classes(value_top, value_middle, value_low, top_regex_n_clicks, middle_regex_n_clicks, low_regex_n_clicks,
                     top_ids, middle_ids, low_ids, table_data, regex_type_top, regex_value_top, regex_type_middle,
                     regex_value_middle, regex_type_low, regex_value_low):
    if value_top is None:
        value_top = [] + top_ids
    if value_middle is None:
        value_middle = [] + middle_ids
    if value_low is None:
        value_low = [] + low_ids

    ctx = dash.callback_context
    if 'top-regex-apply-button' in ctx.triggered[0]['prop_id']:
        if (regex_type_top is None) or ((regex_value_top is None) and ("STR" in regex_type_top)):
            raise PreventUpdate

        top_regex = regex_type_top
        if regex_value_top is not None:
            top_regex = regex_type_top.replace("STR", regex_value_top)

        filtered_top_ids = [row['id']
                            for row in table_data
                            if (not row['id'] in value_top + value_middle + value_low)
                            & (len(re.findall(top_regex, row['Expression'])) > 0)]
        value_top += filtered_top_ids

    elif 'middle-regex-apply-button' in ctx.triggered[0]['prop_id']:
        if (regex_type_middle is None) or ((regex_value_middle is None) and ("STR" in regex_type_middle)):
            raise PreventUpdate

        middle_regex = regex_type_middle
        if regex_value_middle is not None:
            middle_regex = regex_type_middle.replace("STR", regex_value_middle)
        filtered_middle_ids = [row['id']
                               for row in table_data
                               if (not row['id'] in value_top + value_middle + value_low)
                               & (len(re.findall(middle_regex, row['Expression'])) > 0)]
        value_middle += filtered_middle_ids
    elif 'low-regex-apply-button' in ctx.triggered[0]['prop_id']:
        if (regex_type_low is None) or ((regex_value_low is None) and ("STR" in regex_type_low)):
            raise PreventUpdate

        low_regex = regex_type_low
        if regex_value_low is not None:
            low_regex = regex_type_low.replace("STR", regex_value_low)

        filtered_low_ids = [row['id']
                            for row in table_data
                            if (not row['id'] in value_top + value_middle + value_low)
                            & (len(re.findall(low_regex, row['Expression'])) > 0)]
        value_low += filtered_low_ids

    elif (not "class" in ctx.triggered[0]['prop_id']) or (not isinstance(table_data, list)):
        raise PreventUpdate

    top_ids = value_top
    middle_ids = value_middle
    low_ids = value_low

    options = [{"label": f"{row['Expression']} (score {row['Reward']})",
                "value": row['id'],
                "disabled": (row['id'] in top_ids + middle_ids + low_ids)}
               for row in table_data]

    return options, options, options, top_ids, middle_ids, low_ids, value_top, value_middle, value_low


def run_server(port=8050, debug=True):
    app.run_server(port=port, debug=debug)


if __name__ == "__main__":
    run_server()
