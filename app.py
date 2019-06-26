import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import os
import csv
from dash.dependencies import Input, Output
import dash_table
import pickle
import pandas as pd
import numpy as np
import heapq #used in rank matrix
import copy


#---------GET DATA-----------------------------------------------------------------------
interacting_drugs = '/Users/mkocher/Desktop/ARTChat/data/interacting_drug.txt'
interacting_drugs_list = []
with open(interacting_drugs) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=';')
    for row in csv_reader:
        interacting_drugs_list.append(row)


#Pickle Dump:
with open('/Users/mkocher/Desktop/ARTChat/pickled_data/ranks_dist.pkl','rb') as f: ranks_dist = pickle.load(f)
reduced_list = pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/reduced_list.pkl")
principal_df9 = pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/principal_df9.pkl")
dd_compiled_art= pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/dd_compiled_art.pkl")
dd_compiled = pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/dd_compiled.pkl")
cluster_results=pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/cluster_results.pkl")
drugs_art=pd.read_pickle("/Users/mkocher/Desktop/ARTChat/pickled_data/drugs_art.pkl")

drugs_art_app = [
    'Ziagen', 'Emtriva', 'Epivir', 'Viread', 'Retrovir', 'Pifeltro', 'Sustiva',
    'Intelence', 'Viramune', 'Viramune XR', 'Edurant', 'Reyataz', 'Prezista',
    'Lexiva', 'Norvir', 'Invirase', 'Aptivus', 'Fuzeon', 'Selzentry',
    'Tivicay', 'Isentress', 'Isentress HD', 'Trogarzo', 'Tybost', 'Epzicom',
    'Triumeq', 'Trizivir', 'Evotaz', 'Biktarvy', 'Prezcobix', 'Symtuza',
    'Dovato', 'Juluca', 'Delstrigo', 'Atripla', 'Symfi', 'Symfi Lo', 'Genvoya',
    'Stribild', 'Odefsey', 'Complera', 'Descovy', 'Truvada', 'Cimduo',
    'Combivir', 'Kaletra'
]

#List of side-effects -----------------------------------------------------------
side_effects_file = '/Users/mkocher/Desktop/ARTChat/data/side_effects_dash.txt'
side_effect_list = []
with open(side_effects_file) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter = ',')
    for row in csv_reader:
        side_effect_list.append(row) 

#-----------APP WORK--------------------------------------------------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions']=True

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
       
    html.H1(children='ARTChat',style={'textAlign': 'center','color': '#581845'}),
    html.H3(children='HIV Antiretroviral Therapy (ART) Prescription Tool for Physicians & Patients',style={'textAlign':'center','color':'gray'}),

    html.Div(children = 'Let\'s see if ARTChat can help you find the right combination. Enter information below:'),
    html.Br(),

    html.Label('Antiretroviral Therapy Drugs'),    
    dcc.Dropdown(
        id='drug_user',value = 'Genvoya',
        options=[{'label': i,'value' : i} for i in drugs_art_app],
        placeholder='Select your ART Brand Name'),

    html.Label('Side-Effects'),
    dcc.Dropdown(
        id='side_effect_user',
        options=[{'label': i,'value' : i} for i in side_effect_list[0]],value = ['Asthenia'],
        placeholder='Select the side-effects(s) you would like to control',
        multi=True),

    html.Label('Co-administered Drugs'),
    dcc.Dropdown(
        id='coadmin_drug_user',
        options=[{'label': i,'value' : i} for i in interacting_drugs_list[0]],value = ['Amiodarone'],
        placeholder='Select any substances/drugs you might be taking along with your ART',
        multi=True),
#     html.Div(style={'fontColor': 'blue'}, id='drug_info')

    html.Br(),
    html.Div(id='drug_info',style={'color': '#3352FF'}),

    html.H4(id = 'combinations_finals'),
    html.Br(),    

    html.Div(children = 'Safe of administer individual drugs are listed here:'),

    html.Div(id='drugs_recommended_final_1'),
    html.Br(),

    html.Div(children = 'Note: The information offered in this app are recommendations to aid in the selection of alternative regimens. It does not account for drug-resistance information/pregnancy status so use as a supportive tool only.')



])

@app.callback(
    [Output('drugs_recommended_final_1', 'children'),
    Output('drug_info','children'),
    Output('combinations_finals','children')],
    [Input('drug_user', 'value'),
     Input('side_effect_user', 'value'),
     Input('coadmin_drug_user', 'value')])
def compile_figure(drug_user,side_effect_user,coadmin_drug_user):

    print(drug_user)
    print(side_effect_user)
    print(coadmin_drug_user)

    #Recommendation Pipeline:
    drugs_avoid_all=[]

    for i in np.arange(0,len(side_effect_user)):
        drugs_avoid = drugs_to_avoid(drug_name = drug_user, side_effect_user = side_effect_user[i])
        drugs_avoid = drugs_avoid.tolist()
        drugs_avoid_all.append(drugs_avoid) #appends all drugs to avoid

    drugs_avoid_all = [item for sublist in drugs_avoid_all for item in sublist]
    print(drugs_avoid_all)
    drugs_avoid_all = list(dict.fromkeys(drugs_avoid_all))
    print(drugs_avoid_all)

    print('List of drugs to avoid based on contribution to side-effects are {0}'.format(drugs_avoid_all))

    rank_1 = 0
    count_nrti = 0
    count_ii = 0
    count_pi = 0

    drugs_recommended_final = pd.DataFrame(columns=['Brand_Names','Generic_Name/Other_Names','Drug_Class'])

    loop_end = 0

    while((loop_end != 1) & (rank_1 < 3)):

        #Step 2: Cluster Identification:
        drugs_1 = cluster_identifier(drug_name = drugs_avoid_all, rank = rank_1) #returns drugs in the ranked cluster
        print('Drugs in clusters at rank {0} are {1}'.format(rank_1,drugs_1))
        
        
        #Step 3: Frequency Check
        drugs_2 = copy.deepcopy(drugs_1)
        for i in np.arange(len(side_effect_user)): #Going through each side-effect
            drugs_1 = copy.deepcopy(drugs_2)
            drugs_2 = frequency_check(possible_drugs_1 = drugs_1, side_effect = side_effect_user[i])
                      #for each side-effect returning only the drugs that have side-effect % <50%
            drugs_2 = drugs_2.tolist()
        
        drugs_2 = pd.DataFrame({'Drug':drugs_2})#input for interaction checker is a dataframe
        
        #Step 4: Ensuring the drugs don't interact with anything else
        drugs_25 = interaction_checker(drugs_recommended = drugs_2,coadmin_drug_user = coadmin_drug_user)

        #Step 5: What are the different classes of the drugs?
        drugs_3 = class_check(possible_drugs_2 = drugs_25) #input is a dataframe of drugs
        
        drugs_3 = drugs_3.drop_duplicates('Generic_Name/Other_Names') #drop duplicate values

        #Step 6: Confidence value
        count_nrti = count_nrti + len(drugs_3[drugs_3['Drug_Class'] == 'NRTI']) #can double count if it goes to the same cluster twice
        count_ii   = count_ii   + len(drugs_3[drugs_3['Drug_Class'] == 'INSTI'])
        count_pi   = count_pi   + len(drugs_3[drugs_3['Drug_Class'] == 'PI'])
        
        if (count_nrti >= 2):
            if ((count_pi >= 1) | (count_ii >= 1)):
                loop_end = 1 #exit out of the loop 
        
        rank_1 +=1 #move to the next cluster    
        drugs_recommended_final = drugs_recommended_final.append(drugs_3)

    print('We went to the {0} farthest cluster to find combinations'.format(rank_1-1))
    print('All drugs recommended are:')
    print(drugs_recommended_final)
    drugs_recommended_final = drugs_recommended_final.drop_duplicates('Generic_Name/Other_Names')

    data = drugs_recommended_final.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (drugs_recommended_final.columns)]
    avoid_these = 'List of drugs to avoid based on contribution to side-effects are {0}'.format(drugs_avoid_all)

    combinations_final = combination_recommender(drugs_recommended_final)

    if (len(combinations_final) == 0):
        output_combinations = 'No existing brands offer an appropriate combination based on your regimen & functionality of this app'
    else:
        output_combinations = 'The following combinations may be considered: {0}'.format(combinations_final)
        



    return dash_table.DataTable(data=data, columns=columns),avoid_these,output_combinations
    
#------------------FUNCTION DUMP-----------------------------------------------------------------------------------------------------
def drugs_to_avoid(drug_name, side_effect_user):
    '''
    Identify the drugs to avoid based on the side-effect frequency
    '''
    print('------------------------DRUGS TO AVOID--------------------------')
    #1. Finding the location of the drug taken in the HIV/ART database
    drug_taken = drugs_art.loc[drugs_art['Brand_Names'].str.lower() ==
                               drug_name.lower()]

    #2. Checking if it is combination drug
    if drugs_art['Drug_Class'][drug_taken.index[0]] == 'Comb':
        print('Your therapy is a cocktail of the following drugs:')
    else:
        print('The generic name of your drug is:')

    #3. Extracting the ingredients of the combination drugs
    ingr_list = drugs_art['Generic_Name/Other_Names'][drug_taken.index[0]]
    print(ingr_list)  #print statement

    #4. Which drug(s) contributes most to the side-effect in question
    freq_side_effect_user = np.zeros(len(ingr_list))
    for i in np.arange(0, len(ingr_list) - 1):

        #5. First check if the drug is represented in the side-effect database:
        if ((ingr_list[i] in reduced_list.index) == True):
            freq_side_effect_user[i] = reduced_list.loc[ingr_list[i],
                                                  side_effect_user]
        else:
            freq_side_effect_user[
                i] = np.nan  #if the drug doesn't exit in the side-effect database
            #assign NaN to the frequency
            print('{0} does not exist in the SIDER database'.format(
                ingr_list[i]))

    #6. Which drugs contribute most of these side-effects? We will avoid them in the next regimen
    max_freq = np.nanmax(freq_side_effect_user)

    ingr_list = np.asarray(ingr_list)  #converting ingr_list to numpy array
    drugs_avoid = ingr_list[np.where(freq_side_effect_user == max_freq)]

    print(
        'The drug(s) that contribute(s) most to your {0} side-effect is(are): {1}'
        .format(side_effect_user, drugs_avoid))

    return(drugs_avoid)


def cluster_identifier(drug_name, rank):
    '''
    Identifies the cluster based on rank requested and returns drugs in the cluster
    '''
    #Step 1: Finding clusters that the undesirable drugs are seated in
    print('------------------------CLUSTER IDENTIFIER--------------------------')
    cluster_nos=[]
    cluster_desired_all=[]
    for i in np.arange(0,len(drug_name)):

        index = cluster_results.loc[cluster_results['Drug'].str.lower() == drug_name[i].lower()]
        
        cluster_no = index['Cluster_no'].values  #gives you cluster number of the drug
        cluster_no = cluster_no[0]  #collapses numpy array to a number for ease in indexing
        cluster_nos.append(cluster_no) #cluster numbers
                
        #Step 2: Find the cluster at the nth (rank) largest distance
        loc_array = heapq.nlargest(3,range(len(ranks_dist[cluster_no])), key=ranks_dist[cluster_no].__getitem__)
        #above: list of farthest to closest cluster
        cluster_desired = loc_array[rank]
        cluster_desired_all.append(cluster_desired)
    
    #Step 3: Make sure cluster_desired_all has no values from cluster_nos
    cluster_desired_shaved=[]
    for i in np.arange(0,len(cluster_desired_all)):
        if cluster_desired_all[i] not in cluster_nos:
            cluster_desired_shaved.append(cluster_desired_all[i])
        
    #Step 4: Make cluster_desired_shaved into a Series
    cluster_desired_shaved = pd.DataFrame({'desired_cluster':cluster_desired_shaved})
    
    #Step 5: Find the drugs within these desired clusters
    index_dr = pd.merge(left=cluster_results,right=cluster_desired_shaved,left_on='Cluster_no',right_on='desired_cluster')    
    
    print('hola')
    print(index_dr['Drug'].unique)
    return(index_dr['Drug']) #returns a pandas Series
    

def frequency_check(possible_drugs_1,side_effect):
    '''
    This function reduces a list of drug to ones where frequency of undesirable sideeffects < 50%
    '''
    #possible_drugs_1 = possible_drugs_1.tolist() #convert the drug series to a list
    
    drug_frequency_option = reduced_list.loc[possible_drugs_1,side_effect] #
    possible_drugs_2 = drug_frequency_option[drug_frequency_option.values < 50.00]
    return(possible_drugs_2.index)
    
def interaction_checker(drugs_recommended,coadmin_drug_user):

    string_1 = drugs_recommended['Drug'].tolist() #list of drugs that match a frequency criterion
    #string_1 = drugs_recommended
    #Cleaning up columns in Compiled ART and their drug-drug interaction database
    dd_compiled_art['Generic_Name/Other_Names'] = [str(i).replace('[', '').replace(']', '') for i in dd_compiled_art['Generic_Name/Other_Names']]
    dd_compiled_art['Generic_Name/Other_Names'] = [str(i).replace("\'", '') for i in dd_compiled_art['Generic_Name/Other_Names']]

    #Identifying ARTs that interact
    interactions_only = dd_compiled_art.loc[(dd_compiled_art['Generic_Name/Other_Names'].isin(string_1)) & dd_compiled_art['Interacting_Drug'].isin(coadmin_drug_user) & (dd_compiled_art['Management_Code'] == 3.0)]

    to_drop = interactions_only['Generic_Name/Other_Names']
    to_drop_name = to_drop.values.tolist()
    for i in np.arange(0,len(to_drop_name)):
        drugs_recommended = drugs_recommended[drugs_recommended.Drug != to_drop_name[i]]

    return(drugs_recommended) 

def class_check(possible_drugs_2): 
    '''
    returns a confidence value, if desired combination of 2 NRTIs and 1 PI/Insti is achieved
    '''
    #Step 1: Create a dataframe with drugs & their labels 
    drugs_copy = copy.deepcopy(drugs_art)
    drugs_copy['Generic_Name/Other_Names'] = [str(i).replace('[', '').replace(']', '') for i in drugs_copy['Generic_Name/Other_Names']]
    drugs_copy['Generic_Name/Other_Names'] = [str(i).replace("\'", '') for i in drugs_copy['Generic_Name/Other_Names']]
    drugs_labeled = drugs_copy.loc[(drugs_copy['Generic_Name/Other_Names'].isin(possible_drugs_2['Drug'].tolist()))]
    
    return(drugs_labeled)

def combination_recommender(drugs_recommended_final):
    a = drugs_art.groupby('Drug_Class')
    comb_df = a.get_group('Comb').reset_index() 
    comb_rec = []
    for i in np.arange(0,len(comb_df)):
        check = len(comb_df.loc[i]['Generic_Name/Other_Names']) #number of components in each combination    
        truth = drugs_recommended_final['Generic_Name/Other_Names'].isin(comb_df.loc[i]['Generic_Name/Other_Names'])
        check_i = np.sum(truth.values) #number of trues
        if check == check_i:
            comb_rec.append(comb_df.loc[i]['Brand_Names'])
    return(comb_rec)


#------------------------------------------------------------------------------------------------------------------------------------


'''
@app.callback(
    #Output('recommender_drugs', 'value'),
    [Input('brand-names', 'value'),
     Input('side-effects', 'value'),
     Input('co-administered-drugs', 'value')])
'''
    
if __name__ == '__main__':
    app.run_server(debug=True)

#html.Div(id='output-container')    

#def update_output(value):
#    return 'You have selected "{}"'.format(value)
