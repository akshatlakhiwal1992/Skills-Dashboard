#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import pandas as pd
import plotly.graph_objects as go


# In[2]:


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server


# In[3]:


configurations = {
    "2015": {
        "Degree Excluded": {
            "High Wage": ["C*S*Y"],
            "Low Wage": ["~C", "P", "Y"]
        },
        "Degree Included": {
            "High Wage": ["C*S*Y", "C*S*D"],
            "Low Wage": ["~C", "P", "Y", "D"]
        }
    },
    "2016": {
        "Degree Excluded": {
            "High Wage": ["C*S*Y"],
            "Low Wage": ["~C", "P", "Y"]
        },
        "Degree Included": {
            "High Wage": ["C*S*Y", "C*S*D"],
            "Low Wage": ["~C", "P", "Y", "D"]
        }
    },
    "2017": {
        "Degree Excluded": {
            "High Wage": ["C*S*Y"],
            "Low Wage": ["~C", "P", "Y"]
        },
        "Degree Included": {
            "High Wage": ["C*S*Y", "C*S*D"],
            "Low Wage": ["~C", "P", "Y", "S*D", "~S*~D"]
        }
    },
    "2018": {
        "Degree Excluded": {
            "High Wage": ["C*S*Y", "P*C*S"],
            "Low Wage": ["~C", "P", "Y"]
        },
        "Degree Included": {
            "High Wage": ["C*S*D", "P*C*S", "~P*C*Y*~D"],
            "Low Wage": ["~C", "P", "Y", "S*D", "~S*~D"]
        }
    },
    "2019": {
        "Degree Excluded": {
            "High Wage": ["C*S*Y", "P*C*S"],
            "Low Wage": ["~C", "Y", "S*Y"]
        },
        "Degree Included": {
            "High Wage": ["C*S*D", "~S*~D", "~C*~Y*~D", "~P*Y*~D"],
            "Low Wage": ["S*D", "~S*~D", "~C*~S", "P*D"]
        }
    }
}


# In[4]:


# === Mapping of variable codes ===
var_map = {
    "C": "m_fsG_pct_cognitive",
    "P": "m_fsG_pct_physical",
    "S": "m_fsG_pct_sensory",
    "Y": "m_fsG_pct_psychomotor",
    "D": "m_fsG_zavg_degree_cognitive"
}


# In[5]:


# === Layout ===
app.layout = html.Div([
    html.H3("fsQCA Configuration Explorer"),

    html.Label("Select Configuration Type:"),
    dcc.Dropdown(
        id='conf-type-dropdown',
        options=[
            {"label": "Degree Excluded", "value": "Degree Excluded"},
            {"label": "Degree Included", "value": "Degree Included"}
            ],
        value="Degree Excluded",
        style={"width": "50%"}
    ),

    dcc.Graph(id='bubble-chart'),
    dcc.Store(id='selected-config'),

    html.H4("Occupations Matching Configuration"),
    html.Div("Sorted in descending order of match score, which counts how many conditions from the selected configuration are satisfied by the occupation."),
    dash_table.DataTable(id='occupation-table', page_size=10),



    html.H4("Skills for Selected Occupation"),
    dash_table.DataTable(id='skill-table', page_size=200)
])



# In[6]:


# === Callbacks ===
@app.callback(
    Output('bubble-chart', 'figure'),
    Input('conf-type-dropdown', 'value')
)
def update_chart(conf_type):
    all_years = list(configurations.keys())
    high_configs = set()
    low_configs = set()

    # Collect unique configurations across years for each group
    for year in all_years:
        high_configs.update(configurations[year][conf_type]["High Wage"])
        low_configs.update(configurations[year][conf_type]["Low Wage"])

    df_rows = []
    for year in all_years:
        high = configurations[year][conf_type]["High Wage"]
        low = configurations[year][conf_type]["Low Wage"]

        for config in high_configs:
            df_rows.append({
                'year': year,
                'config': config,
                'type': 'High Wage' if config in high else None
            })

        for config in low_configs:
            df_rows.append({
                'year': year,
                'config': config,
                'type': 'Low Wage' if config in low else None
            })

    df = pd.DataFrame(df_rows)

    fig = go.Figure()

    for wage_type, color in zip(["High Wage", "Low Wage"], ["blue", "red"]):
        subset = df[df['type'] == wage_type]
        fig.add_trace(go.Scatter(
            x=subset['year'],
            y=subset['config'],
            mode='markers',
            marker=dict(size=20, color=color),
            name=wage_type,
            customdata=subset[['year', 'config']].values,
            hovertemplate="Year: %{x}<br>Config: %{y}<extra></extra>"
        ))
        
    # Estimate chart height based on number of unique configs
    unique_configs = list(set(df['config']))
    chart_height = max(500, 40 * len(unique_configs))  # 40px per config, min 500px
    
    fig.update_layout(
        height=chart_height,
        xaxis_title="Year",
        yaxis_title="Configuration",
        legend_title="Configuration Type",
        showlegend=True,
        margin=dict(l=120, r=40, t=40, b=40)  # Add extra left margin for long config names
    )


    return fig


@app.callback(
    Output('selected-config', 'data'),
    Input('bubble-chart', 'clickData')
)
def store_selection(clickData):
    if clickData:
        year, label = clickData['points'][0]['customdata']
        return {'year': year, 'label': label}
    return {}

@app.callback(
    Output('occupation-table', 'data'),
    Input('selected-config', 'data'),
    Input('conf-type-dropdown', 'value')
)
def update_occupations(selected, conf_type):
    if not selected:
        return []

    label = selected['label']
    year = selected['year']

    # Parse configuration into variable conditions (True = presence, False = absence)
    conditions = []
    for cond in label.split("*"):
        if cond.startswith("~"):
            conditions.append((var_map[cond[1]], False))
        else:
            conditions.append((var_map[cond], True))

    # Load data (always unadjusted unless you add adjusted mode back later)
    fs_file = f"./fsQCA_demandunadjusted_{year}.csv"
    df = pd.read_csv(fs_file).copy()

    # ✅ STEP 1: True fsQCA configuration membership filter (not match score)
    mask = pd.Series(True, index=df.index)
    for col, include in conditions:
        if include:
            mask &= (df[col] > 0.5)
        else:
            mask &= (df[col] < 0.5)

    df = df[mask].copy()  # Keep only occupations that satisfy the configuration

    # ✅ STEP 2: Now compute match score *as validation only*
    match_scores = []
    for _, row in df.iterrows():
        values = []
        for col, include in conditions:
            val = row[col]  # fuzzy membership value (0–1)
            if include:
                values.append(val)       # presence condition → use membership
            else:
                values.append(1 - val)   # absence condition → use complement
        match_scores.append(sum(values) / len(values))  # mean similarity (0–1)
    
    df['match_score'] = match_scores
    df['avg_salary'] = df['m_fsY2_salary']

    # Sort by salary or match score (since all are full matches, either is fine)
    df = df.sort_values(by='match_score', ascending=False)

    # Load occupation titles
    combined_df = pd.read_csv(f"./combinedSkills_{year}.csv")
    title_map = dict(combined_df[['O*NET-SOC Code', 'Title']].drop_duplicates().values)

    # ✅ Build final output
    results = pd.DataFrame({
        'Occupation Code': df['onetsoccode'],
        'Occupation Title': [title_map.get(code, "") for code in df['onetsoccode']],
        'Match Score': df['match_score'],
        'Average Salary ($)': df['avg_salary']
    })

    return results.to_dict('records')



@app.callback(
    Output('skill-table', 'data'),
    Input('occupation-table', 'active_cell'),
    Input('occupation-table', 'data'),
    Input('selected-config', 'data'),
    Input('conf-type-dropdown', 'value')
)
def update_skills(active_cell, rows, selected, conf_type):
    if active_cell is None or not selected:
        return []

    year = selected['year']
    occ_code = rows[active_cell['row']]['Occupation Code']

    combined_df = pd.read_csv(f"./combinedSkills_{year}.csv")
    skills_df = combined_df[combined_df['O*NET-SOC Code'] == occ_code][['Element Name']].drop_duplicates()

    ability_df = pd.read_excel("./skill-abilityTableOneHot_March21_SingleAbility.xlsx")
    merged = skills_df.merge(
        ability_df[['Skill Name', 'Cognitive', 'Physical', 'Sensory', 'Psychomotor']],
        left_on='Element Name',
        right_on='Skill Name',
        how='left'
    ).drop(columns=['Skill Name']).fillna(0)

    cog_skills = merged[merged['Cognitive'] == 1]['Element Name'].tolist()
    phy_skills = merged[merged['Physical'] == 1]['Element Name'].tolist()
    psy_skills = merged[merged['Psychomotor'] == 1]['Element Name'].tolist()
    sen_skills = merged[merged['Sensory'] == 1]['Element Name'].tolist()

    degree_df = pd.read_csv("./skillLevelNode_Metrics_Panel.csv")
    degree_df = degree_df[(degree_df['Year'] == int(year)) & (degree_df['Adjustment'] == 'Unadjusted')]
    cog_degree_map = degree_df.set_index('Skill')['Degree'].to_dict()

    max_len = max(len(cog_skills), len(phy_skills), len(psy_skills), len(sen_skills))
    cog_skills += [""] * (max_len - len(cog_skills))
    phy_skills += [""] * (max_len - len(phy_skills))
    psy_skills += [""] * (max_len - len(psy_skills))
    sen_skills += [""] * (max_len - len(sen_skills))

        # Summary counts
    total_skills = len(merged)
    counts = {
        "Cognitive": len([s for s in cog_skills if s]),
        "Physical": len([s for s in phy_skills if s]),
        "Psychomotor": len([s for s in psy_skills if s]),
        "Sensory": len([s for s in sen_skills if s])
    }
    percentages = {k: f"{(v / total_skills * 100):.1f}%" for k, v in counts.items()}
    
    # First row: Total counts
    summary_row_1 = {
        "Cognitive Skills": f"Total: {counts['Cognitive']}",
        "Physical Skills": f"Total: {counts['Physical']}",
        "Psychomotor Skills": f"Total: {counts['Psychomotor']}",
        "Sensory Skills": f"Total: {counts['Sensory']}",
        "Cognitive Skill Degree": ""
    }
    
    # Second row: Percentages
    summary_row_2 = {
        "Cognitive Skills": f"%: {percentages['Cognitive']}",
        "Physical Skills": f"%: {percentages['Physical']}",
        "Psychomotor Skills": f"%: {percentages['Psychomotor']}",
        "Sensory Skills": f"%: {percentages['Sensory']}",
        "Cognitive Skill Degree": ""
    }
    
    # Now add actual rows
    result = [summary_row_1, summary_row_2]
    for i in range(max_len):
        cog = cog_skills[i]
        row = {
            "Cognitive Skills": cog,
            "Physical Skills": phy_skills[i],
            "Psychomotor Skills": psy_skills[i],
            "Sensory Skills": sen_skills[i],
            "Cognitive Skill Degree": round(cog_degree_map[cog], 2) if cog in cog_degree_map else ""
        }
        result.append(row)
    
    return result




# In[7]:


if __name__ == '__main__':
    app.run_server(debug=True, port=1010)




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




