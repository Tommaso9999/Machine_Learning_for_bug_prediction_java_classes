# Information Modelling & Analysis: Project 2

Student: Tommaso Verzegnassi 

Please follow the instructions provided in the project slides 
and consider the submission instructions available on iCorsi.

For your convencience, I the following resources are available in the `resources` folder:
- **defects4j-checkout-closure-1f**: The output of the command `defects4j checkout -p Closure -v 1f -w ...`
- **modified_classes** The list of buggy classes in: `framework/projects/Closure/modified_classes/`



1. I have implemented the requested functionalities of phase 1 working on the exactract_feature_vectors.py file. After some adjustments the results I get for all the metrics of PeepHoleSimplifyRegExp and MinimizeExitPoints match the provided guideline. you can see the output csv of this phase in feature_vectors/feature_vectors_not_labeled.csv relative path. The first two rows store the guideline classes info if you want to check (results should also be printed when you run the script).


2. the script label_feature_vectors introduces a new column named "buggy" to the csv file created in the previous script. The value 1 is assigned to the classes that match the given criteria, 0 otherwsie. This new csv file is stored in the following relative path: feature_vectors/feature_vectors_labeled.csv. 
 
3. the script train_classifiers applies all the algorithms with the manually identified best hyperparameters configuration. you can find results for other configurations in the report or material_for_report subfolder. 

4. for this final phase I have worked on evaluate classifiers.py and implemented the requested analysis on the classifiers results. It seems like most of them, when considering the F score value, perform slightly better than the biased classifier with the parameters I have used. 