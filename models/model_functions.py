import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

heat_fringe =  LinearSegmentedColormap.from_list("", ['gold', 'darkorange', '#C43714'], N=256, gamma=1.0)
bar_colors = ['navy', 'lightskyblue']

def train_model(ModelClass, X_train, Y_train, **kwargs):
    '''
        function trains a model with Train data and gives back this model
    '''
    model = ModelClass(**kwargs)
    model.fit(X_train, Y_train)

    return model

def model_scores_df(model, Train_X, Test_X, Train_Y, Test_Y, model_name:str):
    '''
        the function predict Train and Test Data and caclculate the 
        accuracy, recall and precision
        gives back a dictionary with all the metrics
    '''
    pred_train = model.predict(Train_X)
    pred_test = model.predict(Test_X)
    model_df = pd.DataFrame([{'model_name': model_name,  
                'train_accuracy': accuracy_score(Train_Y, pred_train).round(2), 
                'test_accuracy': accuracy_score(Test_Y, pred_test).round(2),
                'train_precision': precision_score(Train_Y, pred_train).round(2), 
                'test_precision': precision_score(Test_Y, pred_test).round(2),
                'train_recall': recall_score(Train_Y, pred_train).round(2), 
                'test_recall': recall_score(Test_Y, pred_test).round(2),
                'train_f1': f1_score(Train_Y, pred_train).round(2), 
                'test_f1': f1_score(Test_Y, pred_test).round(2),
                }])

    return model_df

def conf_matrix_heatmap_perc(confusion_matrix_local):
    conf_matrix_perc = (confusion_matrix_local / confusion_matrix_local.sum()) * 100
    h = sns.heatmap(conf_matrix_perc,annot= True, fmt=".1f", cmap= heat_fringe, annot_kws={"size": 12})
    h.tick_params(left = False, bottom = False)

    return

def conf_matrix_heatmap_abs(confusion_matrix_local):
    h = sns.heatmap(confusion_matrix_local,annot= True, fmt=".0f", cmap= heat_fringe, annot_kws={"size": 12})
    h.tick_params(left = False, bottom = False)

    return

def conf_matrix_as_bar_perc(confusion_matrix_local):
    """
    function creates a plot with True and False predicted Values as bar
    out of a confusion matrix with percent value

    Args:
        confusion_matrix_local (_type_): _description_
    """
    conf_matrix_perc = (confusion_matrix_local / confusion_matrix_local.sum()) * 100
    local_df = pd.DataFrame([{'class':'0',
                                   'statement': True,
                                   'pred': conf_matrix_perc[0][0]},
                                   {'class':'0',
                                   'statement': False,
                                   'pred':conf_matrix_perc[0][1] },
                                   {'class':'1',
                                   'statement': True,
                                   'pred':conf_matrix_perc[1][1] },
                                   {'class':'1',
                                   'statement': False,
                                   'pred':conf_matrix_perc[1][0] }])
    
    sns.set_palette(bar_colors)

    b = sns.catplot(data=local_df, x='class',y = 'pred', hue = 'statement', kind='bar')

    b.set(xlabel='',
        ylabel='% of predicted Values',
        title='')
    sns.move_legend(b, "upper right", title=None,  bbox_to_anchor=(0.9, 0.9))
    ax = b.facet_axis(0,0)
    b.set_xticklabels(['No Bank Account', 'Bank Account'])
    for c in ax.containers:
        ax.bar_label(c,fmt ='{:.1f}%' , label_type='edge')

    return

def conf_matrix_as_bar_abs(confusion_matrix_local):
    """
    function creates a plot with True and False predicted Values as bar
    out of a confusion matrix with percent value

    Args:
        confusion_matrix_local (_type_): _description_
    """
    conf_matrix_perc = (confusion_matrix_local / confusion_matrix_local.sum()) * 100
    local_df = pd.DataFrame([{'class':'0',
                                   'statement': True,
                                   'pred': confusion_matrix_local[0][0]},
                                   {'class':'0',
                                   'statement': False,
                                   'pred':confusion_matrix_local[0][1] },
                                   {'class':'1',
                                   'statement': True,
                                   'pred':confusion_matrix_local[1][1] },
                                   {'class':'1',
                                   'statement': False,
                                   'pred':confusion_matrix_local[1][0] }])

    sns.set_palette(bar_colors)

    b = sns.catplot(data=local_df, x='class',y = 'pred', hue = 'statement', kind='bar')

    b.set(xlabel='',
        ylabel='Number of predicted Values',
        title='')
    sns.move_legend(b, "upper right", title=None,  bbox_to_anchor=(0.9, 0.9))
    ax = b.facet_axis(0,0)
    b.set_xticklabels(['No Bank Account', 'Bank Account'])
    for c in ax.containers:
        ax.bar_label(c,fmt ='{:.0f}' , label_type='edge')

    return