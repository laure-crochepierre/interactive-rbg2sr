# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import json
import psutil
import signal
from sympy.parsing.sympy_parser import (parse_expr, function_exponentiation, implicit_application,
                                        implicit_multiplication, standard_transformations)
transformations = standard_transformations + (function_exponentiation,implicit_application, implicit_multiplication)
import time
import pickle
import subprocess


import numpy as np

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
from dash import dash_table
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"

import plotly.graph_objs as go

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
metrics = {"MSE": mean_squared_error, "R2": r2_score,
           "Pearson": lambda y, yhat: pearsonr(y, yhat)[0],
           "Spearman": lambda y, yhat: spearmanr(y, yhat)[0],
           }

waiter = [dbc.Container([dbc.Row(dbc.Col(dbc.Spinner(show_initially=True, color="secondary"), width={"size": 6, "offset": 6})),
                  dbc.Row(dbc.Col(html.H4("Waiting to receive pairs of expressions to compare"), #"En attente de réception des paires d'expressions à comparer"),
                                  width={"size": 6, "offset": 4}))],
    className="p-3 bg-light rounded-3")]

# Launch app
app = dash.Dash('ipref', external_stylesheets=[dbc.themes.MORPH, FONT_AWESOME])   # replaces dash.Dash
app.layout = html.Div([
    dbc.Row([dbc.Col([
        html.H1("Reinforcement Based Grammar Guided Symbolic Regression with Preference Learning",
                style={'margin': 25, 'textAlign': 'center'}),
    ], width={'size':10}),
        dbc.Col([dbc.Button("New training", # 'Nouvel entrainement',
                            id="new_training", outline=True,
                            color="primary", style={'margin': 30})])
        ]),
    dcc.Store(id='local-gui-data-logdir', storage_type='local'),
    dcc.Store(id='local-current-step', storage_type='local', data=0),
    dcc.Store(id='local-pid', storage_type='local'),
    dcc.Store(id='local-id-pairs', storage_type='local'),
    dcc.Store(id='local-pair-indexes', storage_type='local'),
    dcc.Store(id='local-grammar', storage_type='local'),
    html.Div([dbc.Row(dbc.Col(dbc.Button("Start Training", #"Commencer l'entrainement",
                                         id="launch-training", n_clicks=0,
                                         className="d-grid col-12 mx-auto",
                                         outline=True,
                                         color="primary", size="lg"),
                              width={"size": 6, "offset": 3}))
              ],  className="gap-2", id="before-training", hidden=True),
    html.Div([dcc.Interval(id="interval-during-training", interval=5*1000, disabled=True),
              html.Div(waiter, id='waiter'),
              html.Div([
                  dcc.Tabs([
                      dcc.Tab(label="Preference pairs", #'Paires de préférences',
                              children=[
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
                                  ], style={"padding-top": "2%"})]),
                      dcc.Tab(label="High-Medium-Low preferences", #'Préférences Hautes-Moyennes-Basses',
                              children=[dbc.Row(dbc.Col(html.Div(id='preference-classes')))]),
                      dcc.Tab(label="Solution suggestion", #"Suggestion d'une solution",
                              children=[dbc.Row(dbc.Col(html.Div(id='solution-suggestion')), style={"padding-top": "2%"})])
                  ]),
              dbc.Row(dbc.Col(html.Div(id='valider-et-continuer'), width={"size": 8, "offset": 2})),
              ], id="iteration_data")],
             hidden=True, id="during-training", style={'margin': 25})
])


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
               ],
              [Input('launch-training', 'n_clicks'),
               Input('interval-during-training', 'n_intervals'),
               Input({'type': 'validate', 'index': ALL}, 'n_clicks'),
               Input({'type': 'delete_pair', 'index': ALL}, 'n_clicks'),
               Input("datatable", "selected_row_ids"),
               Input('new_training', "n_clicks")],
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
               ]
              )
def content_callback(launch_n_clicks, n_intervals, validate_n_clicks, delete_pair_n_clicks, selected_row_indices, new,
                     logdir, current_step, pid, pair_indexes, grammar, local_expression_data, table_data,
                     children, suggestion_box, pref_classes, continuer_box, right_pref, left_pref, both_pref, none_pref,
                     top_idss, middle_idss, low_idss):
    hidden_during = dash.no_update
    hidden_before = dash.no_update
    hidden_waiter = dash.no_update
    hidden_iteration_data = dash.no_update
    interval_disabled = dash.no_update

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
               continuer_box, selected_row_indices

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
        return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
               pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes,\
               continuer_box, selected_row_indices

    elif ctx.triggered[0]['prop_id'] == "launch-training.n_clicks":
        interval_disabled, hidden_during, hidden_before, logdir, pid = callback_launch()
        current_step = 0
        hidden_iteration_data = True
    elif "validate" in ctx.triggered[0]['prop_id']:
        validate_prefs(logdir, pair_indexes, current_step, right_pref, left_pref, both_pref, none_pref,
                       top_ids, middle_ids, low_ids, local_expression_data)
        interval_disabled = False
        hidden_iteration_data = True
        hidden_waiter = False
        children = []
        pair_indexes = None
    elif "delete_pair" in ctx.triggered[0]['prop_id']:
        pair_id_to_drop = json.loads(ctx.triggered[0]['prop_id'].replace('.n_clicks', ''))['index']
        pair_indexes.pop([str(p) for p in pair_indexes].index(pair_id_to_drop))
        children, continuer_box, grammar, table_data, pair_indexes, \
        interval_disabled, hidden_during, hidden_before, current_step = pairs_plot_callback(
            n_intervals, logdir, current_step, pair_indexes, grammar)

    elif "datatable.selected_row_ids" in ctx.triggered[0]['prop_id']:
        new_pair = table_callback(selected_row_indices)
        if new_pair is not None:
            selected_row_indices = []
            pair_indexes = new_pair + pair_indexes
            children, continuer_box, grammar, table_data, pair_indexes, \
            interval_disabled, hidden_during, hidden_before, current_step = pairs_plot_callback(
                n_intervals, logdir, current_step, pair_indexes, grammar)
    elif (pid is not None) and (pid > 0) and psutil.pid_exists(pid):
        if (not pair_indexes is None) and (len(pair_indexes) == 0):
            pair_indexes = None
        children, continuer_box, grammar, table_data, pair_indexes, \
        interval_disabled, hidden_during, hidden_before, current_step = pairs_plot_callback(
            n_intervals, logdir, current_step, pair_indexes, grammar)
        pref_classes = get_classes(table_data, current_step, top_ids, middle_ids, low_ids)

        if not isinstance(grammar, dict):
            return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
           pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes,\
            continuer_box, selected_row_indices
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
        suggestion_box = dbc.Container([
            dbc.Row(
                dbc.Col(
                    html.Div(
                        html.Div([html.P( "Choose an expression to compare with"),  #"Choisir une expression avec laquelle se comparer"),
                                  dcc.Dropdown(id={'type': "suggestion-id", 'index': current_step},
                                               options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                                         "value": row['id']} for row in table_data]),
                                  html.P(" ")], id={'type': "validation-div", 'index': current_step})
                    )
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div([html.P("Suggest an expression"), #"Suggestion d'une expression"),
                              dbc.Alert("Expression: " + expression, color="light",
                                        id={'type': "expression", 'index': current_step})],
                             id={'type': "alert-div", 'index': current_step})
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div([
                        dcc.Store(id={'type': "local-expression-data", 'index': current_step}, storage_type='local',
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
                    dbc.Button("Start a new suggestion", # "Commencer une nouvelle suggestion",
                               id={'type': 'restart-suggest',
                                   'index': current_step},
                               outline=True,
                               color="info",
                               className="d-grid gap-2 col-6 mx-auto",
                               style={"margin": "1%"})
                ),
                dbc.Col(
                    dbc.Button("Suggestion validation", # "Valider la suggestion",
                               id={'type': 'suggestion-validation',
                                   'index': current_step},
                               outline=True,
                               color="info",
                               className="d-grid gap-2 col-6 mx-auto",
                               style={"margin": "1%"}, disabled=True)),
            ]),
            dbc.Row([
                dbc.Col([html.Div(id={"type": "validated-suggestion-div", "index": current_step})])
            ])
        ], className="p-3 bg-light rounded-3")

        if isinstance(children, list) and (len(children) > 0):
            hidden_during = False
            hidden_iteration_data = False
            hidden_waiter = True
    elif pid is None:
        hidden_before = False
        return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
           pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
               continuer_box, selected_row_indices

    return interval_disabled, hidden_during, hidden_before, hidden_waiter, hidden_iteration_data, logdir, pid, \
           pair_indexes, current_step, grammar, table_data, children, suggestion_box, pref_classes, \
        continuer_box, selected_row_indices


def callback_launch():
    os.makedirs("../results/interactive_runs", exist_ok=True)
    writer_logdir = f"../results/interactive_runs/{time.time()}"
    proc = subprocess.Popen([f'python app_interactive_algorithm.py {writer_logdir}'], shell=True)
    print("Training Launched !")
    gui_data_logdir = os.path.join(writer_logdir, 'gui_data')
    os.makedirs(gui_data_logdir, exist_ok=True)
    return False, False, True, gui_data_logdir, proc.pid


def expression_formating(t):
    if (t == '') or ('<' in t):
        return ''

    t = t.replace('np.', '')
    for i in range(10):
        t = t.replace(f'x[:,{i}]', f"x{i}")
    t = parse_expr(t).__repr__()
    return t


def pairs_plot_callback(n_intervals, gui_data_logdir, current_step, combinaisons=None, grammar=None):
    if len(os.listdir(gui_data_logdir)) == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    input_pkls = [os.path.join(gui_data_logdir, f) for f in os.listdir(gui_data_logdir)
                       if not "answers" in f]
    answers_pkls = [os.path.join(gui_data_logdir, f) for f in os.listdir(gui_data_logdir)
            if "answers" in f]

    if len(input_pkls)-len(answers_pkls) == 0:
        return waiter, [], grammar, [], [], False, False, True, current_step
    else:
        current_step = max([int(f.split('/')[-1].replace(".pkl", "")) for f in input_pkls])

    pairs_data_path = os.path.join(gui_data_logdir, f"{current_step}.pkl")

    pairs_data = pickle.load(open(pairs_data_path, 'rb'))
    if combinaisons is None:
        combinaisons = [list(p) for p in pairs_data['combinaisons']]
    rewards = pairs_data['rewards']
    translations = pairs_data['translations']
    y_pred = pairs_data['predicted_values']
    x = pairs_data['x']
    y = pairs_data['target_values']
    top_indices = pairs_data['top_indices'].T[0]
    if (current_step == 1) & (grammar is None):
        grammar = pairs_data['grammar']
    top_expressions = sorted(list(set([translations[i] for i in top_indices])),
                                  key=lambda t: rewards[translations.index(t)], reverse=True)

    table_data = [{'id': translations.index(t),
                   'Expression': expression_formating(t).replace('*', ' * '),
                  'Reward': round(rewards[translations.index(t)], 2)}
                 for t in top_expressions]

    children_blocks = [html.H4(f"Iteration n°{current_step}", style={"textAlign":'center'})]
    for i_pair, pair_ids in enumerate(combinaisons):
        id_left, id_right = pair_ids
        pair_ids = str(pair_ids)
        r_left, r_right = round(rewards[id_left], 3), round(rewards[id_right], 3)
        t_left = expression_formating(translations[id_left])
        t_right = expression_formating(translations[id_right])

        left_fig = go.Figure(data=[go.Scatter(x=x, y=y_pred[id_left], name="y_pred", mode='markers'),
                                   go.Scatter(x=x, y=y, name="y", mode='markers')],
                             layout=go.Layout(xaxis_title="Variable x",
                                              margin=go.layout.Margin(l=1, r=1, b=1, t=1)))

        left_fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        toast_content_left = [
            dbc.Row(dbc.Col(dcc.Graph(figure=left_fig)
                            )
                    ),
            #dbc.Row(dbc.Col([f"Score de récompense : ", html.Strong(r_left)])),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id=f'table-results-left-{i_pair}',
                columns=([{'id': 'Reward', 'name': 'Reward'}] + [{'id': m, 'name': m} for m in metrics.keys()]),
                data=[dict(Reward=r_left, **{name: round(m(y, y_pred[id_left]), 2) for name, m in metrics.items()})],
                editable=False
            ))),
            dbc.Row(dbc.Col(dbc.Button("Choose", #"Choisir",
                                       outline=True, color="secondary", className="d-grid col-12 mx-auto",
                                       id={"type": 'prefer_left', "index": pair_ids}), width={"size": 3, "offset": 5}))]

        right_fig = go.Figure(data=[go.Scatter(x=x, y=y_pred[id_right], name="y_pred", mode='markers'),
                                    go.Scatter(x=x, y=y, name="y", mode='markers')],
                              layout=go.Layout(xaxis_title="Variable x",
                                              margin=go.layout.Margin(l=0, r=0, b=0, t=0)))
        right_fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        toast_content_right = [
            dbc.Row(dbc.Col(dcc.Graph(figure=right_fig))),
            #dbc.Row(dbc.Col([f"Score de récompense : ", html.Strong(r_right)])),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id='table-editing-simple',
                columns=([{'id': 'Reward', 'name': 'Reward'}] + [{'id': m, 'name': m} for m in metrics.keys()]),
                data=[dict(Reward=r_right, **{name: round(m(y, y_pred[id_right]), 2) for name, m in metrics.items()})],
                editable=False
            ))),
            dbc.Row(dbc.Col(dbc.Button("Choose", #"Choisir",
                                       outline=True,
                                       color="secondary",
                                       className="d-grid col-12 mx-auto",
                                       id={"type": 'prefer_right', "index": pair_ids}),
                            width={"size": 3, "offset": 4}))]
        left_toast = dbc.Toast(toast_content_left,
                               header=t_left,
                               style={"width": "100%", "max-width": "48%", "margin": "1%", "font-size": "1rem"})
        right_toast = dbc.Toast(toast_content_right,
                                header=t_right,
                                style={"width": "100%", "max-width": "48%", "margin": "1%", "font-size": "1rem"})
        pair_row = dbc.Form([dbc.Row([left_toast, right_toast]),
                    dbc.Row(dbc.Col(dbc.Button("Equivalent",
                                               outline=True,
                                               color="secondary",
                                               className="d-grid gap-2 col-6 mx-auto",
                                               style={"margin": "1%"},
                                               id={"type": 'prefer_both', "index": pair_ids}),
                                    width={"size": 8, "offset": 2})),
                    dbc.Row(dbc.Col(dbc.Button("Neither", #"Je n'aime aucun des 2",
                                               outline=True,
                                               color="primary",
                                               className="d-grid gap-2 col-6 mx-auto",
                                               style={"margin": "1%"},
                                               id={"type": 'prefer_none', "index": pair_ids}),
                                    width={"size": 8, "offset": 2}))
                    ])
        card_header = dbc.Row([dbc.Col([html.H4(f'Pair {pair_ids}')], width={"size": 10}),
                              dbc.Col([dbc.Button(html.I(className="far fa-times-circle fa-lg",
                                                         **{'aria-hidden': 'true'},
                                                         children=None), id={'type': 'delete_pair', 'index': pair_ids})])])
        children_blocks.append(dbc.Card([dbc.CardHeader([card_header],
                                                        style={"text-align": "center"}),
                                         dbc.CardBody(pair_row)]))
    continuer_box = dbc.Button("Validate and continue", #"Valider et continuer",
                               id={'type': 'validate',
                                   'index': current_step},
                               outline=True,
                               color="warning",
                               className="d-grid gap-2 col-6 mx-auto",
                               style={"margin": "1%"})

    return children_blocks, continuer_box, grammar, table_data, combinaisons, True, False, True, current_step


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
        local_expression_data['queue'] = []
        local_expression_data['translation'] = grammar['start_symbol']
        local_expression_data['current_symbol'] = grammar['start_symbol']
        local_expression_data['action_ids'] = []
        local_expression_data['comparison_with_id'] = None
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
            options=options),
        return dd, local_expression_data, expression, True, []
    elif "suggestion-validation" in ctx.triggered[0]['prop_id']:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               [str(n_clicks_suggestion_validation)+str(local_expression_data)]
    elif "suggestion-id" in ctx.triggered[0]['prop_id']:
        local_expression_data['comparison_with_id'] = suggestion_id
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
        return dash.no_update
    colors = ["secondary", "secondary", "secondary", "secondary"]
    n_clicks = [None, None, None, None]
    if "right" in ctx.triggered[0]['prop_id']:
        colors[0] = "primary"
        n_clicks[0] = 1
    elif "left" in ctx.triggered[0]['prop_id']:
        colors[1] = "primary"
        n_clicks[1] = 1
    elif "both" in ctx.triggered[0]['prop_id']:
        colors[2] = "primary"
        n_clicks[2] = 1
    elif "none" in ctx.triggered[0]['prop_id']:
        colors[3] = "primary"
        n_clicks[3] = 1
    return colors+n_clicks


def validate_prefs(logdir, pair_indexes, current_step, right_pref, left_pref, both_pref, none_pref,
                   top_ids, middle_ids, low_ids, suggestions):
    right_pref = [{True: '', False: 'r'}[answer is None] for answer in right_pref]
    left_pref = [{True: '', False: 'l'}[answer is None] for answer in left_pref]
    both_pref = [{True: '', False: 'b'}[answer is None] for answer in both_pref]
    none_pref = [{True: '', False: 'n'}[answer is None] for answer in none_pref]
    prefs = {"pairs": {"ids": pair_indexes,
                       "answers": [r+l+b+n for r, l, b, n in zip(right_pref, left_pref, both_pref, none_pref)]},
             "classes": {"top": top_ids, "middle": middle_ids, "low": low_ids},
             "suggest": [s for s in suggestions if ("<" not in s['translation']) and
                         (s['comparison_with_id'] is not None)]}
    pickle.dump(prefs, open(os.path.join(logdir, f"{current_step}_answers.pkl"), 'wb'))


def table_callback(selected_row_indices):
    if len(selected_row_indices) == 2:
        return [selected_row_indices]
    else:
        return None


def get_classes(table_data, current_step, top_ids, middle_ids, low_ids):
    if not isinstance(table_data, list):
        return dash.no_update
    classes_ranking = [dbc.Col([
        dbc.Row([dbc.Col(html.H4("Best expressions")),  # "Meilleures expressions")),
                 dbc.Col(html.H4("Average expressions")),  # "Expressions moyennes")),
                 dbc.Col(html.H4("Bad expressions"))]),  # "Expression mauvaises"))]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(id={'type': "top-class", 'index': current_step},
                             multi=True,
                             searchable=True,
                             clearable=False,
                             options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                       "value": row['id']} for row in table_data],
                             value=top_ids)
            ]),
            dbc.Col([
                dcc.Dropdown(id={'type': "middle-class", 'index': current_step},
                             multi=True,
                             searchable=True,
                             clearable=False,
                             options=[{"label": f"{row['Expression']} (score {row['Reward']})",
                                       "value": row['id']} for row in table_data],
                             value=middle_ids)
                ]),
            dbc.Col([
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
        dcc.Store(id={'type': 'top-expression-ids', 'index': current_step}, storage_type="local", data=[]),
        dcc.Store(id={'type': 'middle-expression-ids', 'index': current_step}, storage_type="local", data=[]),
        dcc.Store(id={'type': 'low-expression-ids', 'index': current_step}, storage_type="local", data=[])]
    return classes_ranking

@app.callback(
    [Output({'type': 'top-class', 'index': MATCH}, 'options'),
     Output({'type': 'middle-class', 'index': MATCH}, 'options'),
     Output({'type': 'low-class', 'index': MATCH}, 'options'),
     Output({'type': 'top-expression-ids', 'index': MATCH}, 'data'),
     Output({'type': 'middle-expression-ids', 'index': MATCH}, 'data'),
     Output({'type': 'low-expression-ids', 'index': MATCH}, 'data'),
     ],
    [Input({'type': 'top-class', 'index': MATCH}, 'value'),
     Input({'type': 'middle-class', 'index': MATCH}, 'value'),
     Input({'type': 'low-class', 'index': MATCH}, 'value')
     ],
    [State({'type': 'top-expression-ids', 'index': MATCH}, 'data'),
     State({'type': 'middle-expression-ids', 'index': MATCH}, 'data'),
     State({'type': 'low-expression-ids', 'index': MATCH}, 'data'),
     State("datatable", "data")]
)
def pairs_by_classes(value_top, value_middle, value_low, top_ids, middle_ids, low_ids, table_data):

    ctx = dash.callback_context
    if (not "class" in ctx.triggered[0]['prop_id']) or (not isinstance(table_data, list)):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if value_top is None:
        value_top = []
    if value_middle is None:
        value_middle = []
    if value_low is None:
        value_low = []
    top_ids = value_top
    middle_ids = value_middle
    low_ids = value_low

    options = [{"label": f"{row['Expression']} (score {row['Reward']})",
                "value": row['id'],
                "disabled": (row['id'] in top_ids+middle_ids+low_ids)}
               for row in table_data]

    return options, options, options, top_ids, middle_ids, low_ids


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8050, debug=True)

