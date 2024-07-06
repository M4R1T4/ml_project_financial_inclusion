import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score


heat_fringe =  LinearSegmentedColormap.from_list("", ['gold', 'darkorange', '#C43714'], N=256, gamma=1.0)
bar_colors = ['navy', '#DA3287']
sns.set_style('whitegrid', {'grid.linestyle': '--'})

def train_model(ModelClass, X_train, Y_train, **kwargs):
    '''
        function trains a model with Train data and gives back this model
    '''
    model = ModelClass(**kwargs)
    model.fit(X_train, Y_train)

    return model

def knn_cross_validation(X_train, y_train):
    # Fit and evaluate model without hyperparameter tuning using cross validation and unscaled data 
    knn_classifier = KNeighborsClassifier()
    scores = cross_val_score(knn_classifier, X_train, y_train, cv=7, n_jobs=-1)

    # Evaluation 
    print('Score:', round(scores.mean(), 4))
    # plotting the scores and average score
    plt.axhline(y=scores.mean(), color='y', linestyle='-')
    sns.barplot(x=[1,2,3,4,5,6,7],y=scores).set_title('Scores of the K-Folds Models')

    return

def model_scores_df(model, Train_X, Test_X, Train_Y, Test_Y, model_name:str):
    '''
        the function predict Train and Test Data and caclculate the 
        accuracy, recall and precision
        gives back a dictionary with all the metrics
    '''
    pred_train = model.predict(Train_X)
    pred_test = model.predict(Test_X)
    
    """
    model_df = pd.DataFrame([{'model_name': model_name,  
                'train_accuracy': accuracy_score(Train_Y, pred_train).round(2), 
                'test_accuracy': accuracy_score(Test_Y, pred_test).round(2),
                'train_gmean': geometric_mean_score(Train_Y, pred_train).round(2), 
                'test_gmean': geometric_mean_score(Test_Y, pred_test).round(2),
                'train_sensitivity': sensitivity_score(Train_Y, pred_train).round(2), 
                'test_sensitivity': sensitivity_score(Test_Y, pred_test).round(2),
                'train_specificity': specificity_score(Train_Y, pred_train).round(2), 
                'test_specificity': specificity_score(Test_Y, pred_test).round(2)
                }])
                """
    model_dict = {'model_name': model_name,  
                'train_accuracy': accuracy_score(Train_Y, pred_train).round(2), 
                'test_accuracy': accuracy_score(Test_Y, pred_test).round(2),
                'train_gmean': geometric_mean_score(Train_Y, pred_train).round(2), 
                'test_gmean': geometric_mean_score(Test_Y, pred_test).round(2),
                'train_sensitivity': sensitivity_score(Train_Y, pred_train).round(2), 
                'test_sensitivity': sensitivity_score(Test_Y, pred_test).round(2),
                'train_specificity': specificity_score(Train_Y, pred_train).round(2), 
                'test_specificity': specificity_score(Test_Y, pred_test).round(2)
                }
                

    return model_dict

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

def metrics_line_scatterplot(metric_dict):
    """
    creates a special metrics plot out of the metrics dictionary

    Args:
        metric_dict (_type_): _description_
    """
    local_metric_df = pd.DataFrame([metric_dict])
    g = sns.lineplot(x = [0,1], y = [0,1], color = 'lightgrey')
    g = sns.scatterplot(x = local_metric_df.train_accuracy, y = local_metric_df.test_accuracy, color = '#076B00', s=48)
    g = sns.scatterplot(x = local_metric_df.train_sensitivity, y = local_metric_df.test_sensitivity, color = 'darkorange',  s=48)
    g = sns.scatterplot(x = local_metric_df.train_specificity, y = local_metric_df.test_specificity, color = '#C43714',  s=48)
    g = sns.scatterplot(x = local_metric_df.train_gmean, y = local_metric_df.test_gmean, color = 'Cornflowerblue',  s=48)
    g.set_title(local_metric_df.model_name[0])
    g.set_xlabel('train metrics')
    g.set_ylabel('test metrics')
    g.legend(loc='lower right', 
             #bbox_to_anchor=(0.5, 0.5),
             labels=['proportion 0.5',
                    '',
                    'accuracy',
                    'sensitivity',
                    'specificity',
                    'gmean',
             ])
    plt.yticks(list(np.arange(0,1.1,0.1).round(1)))

    return

def metrics_comp_scatterplot(df_name):
    '''
        function shows a seaborn scatterplot from a special data frame 
        with accuracy, sensitivity, specificity and gmean from train and test data 
    '''

    plt.figure(figsize=(12,8))
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'test_accuracy', color = '#076B00', s=48)
    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'train_accuracy', marker='+', color = '#076B00', s=48)

    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'test_sensitivity', color = 'darkorange', s=48)
    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'train_sensitivity', marker='+', color = 'darkorange', s=48)

    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'test_specificity', color = '#C43714', s=48)
    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'train_specificity', marker='+', color = '#C43714', s=48)

    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'test_gmean', color = 'Cornflowerblue', s=48)
    g = sns.scatterplot(data = df_name, x = 'model_name', y = 'train_gmean', marker='+', color = 'Cornflowerblue', s=48)
    
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(title = 'metrics on train and test data')
    g.legend(loc='upper right', 
             bbox_to_anchor=(1.2, 1),
             labels=['test_accuracy',
                'train_accuracy',
                'test_sensitivity',
                'train_sensitivity',
                'test_specificity',
                'train_specificity',
                'test_gmean',
                'train_gmean'
                ] )
    
    plt.yticks(list(np.arange(0,1.1,0.1).round(1)))
    plt.show();

    return