
#this script was made just to feed the best parameters faster with some copy and paste without needing to type the whole thing


def modify_string(input_string):
   
    modified_string = input_string.replace("'", "")
    modified_string = modified_string.replace(":", "=")
    modified_string = modified_string.replace("{", "").replace("}", "")
    
    return modified_string


modified_str = modify_string("{'memory': None, 'steps': [('standardscaler', StandardScaler()), ('mlpclassifier', MLPClassifier(max_iter=2000))], 'verbose': False, 'standardscaler': StandardScaler(), 'mlpclassifier': MLPClassifier(max_iter=2000), 'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'mlpclassifier__activation': 'relu', 'mlpclassifier__alpha': 0.0001, 'mlpclassifier__batch_size': 'auto', 'mlpclassifier__beta_1': 0.9, 'mlpclassifier__beta_2': 0.999, 'mlpclassifier__early_stopping': False, 'mlpclassifier__epsilon': 1e-08, 'mlpclassifier__hidden_layer_sizes': (100,), 'mlpclassifier__learning_rate': 'constant', 'mlpclassifier__learning_rate_init': 0.001, 'mlpclassifier__max_fun': 15000, 'mlpclassifier__max_iter': 2000, 'mlpclassifier__momentum': 0.9, 'mlpclassifier__n_iter_no_change': 10, 'mlpclassifier__nesterovs_momentum': True, 'mlpclassifier__power_t': 0.5, 'mlpclassifier__random_state': None, 'mlpclassifier__shuffle': True, 'mlpclassifier__solver': 'adam', 'mlpclassifier__tol': 0.0001, 'mlpclassifier__validation_fraction': 0.1, 'mlpclassifier__verbose': False, 'mlpclassifier__warm_start': False}"
)
print(modified_str)
