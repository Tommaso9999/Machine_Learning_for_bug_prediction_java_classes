import os
import re
import pandas as pd
import javalang 


def extract_features(folder_name):
    class_info = []
    class_asts= []

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for root, dirs, files in os.walk("resources/defects4j-checkout-closure-1f/src/com/google/javascript/jscomp"):
        for file in files:

            name=file.split(".")[0]

            if file.endswith(".java"):
                tree=""
                class_name = re.sub("\.java$", "", file)
                class_path = os.path.join(root, file)
              

                with open(class_path, "r") as f:
                    content = f.read()

                try:
                    tree = javalang.parse.parse(content)
                    class_asts.append(tree)
                
                except javalang.parser.JavaSyntaxError:
                    pass


                is_outermost_class_var = True
               
                if is_outermost_class_var:


                    num_methods = 0
                    num_public_methods = 0
                    sum_of_letters_method_names=0

                    for _, node in tree.filter(javalang.tree.MethodDeclaration):
                    
                            num_methods += 1
                            length = len(node.name)
                            sum_of_letters_method_names += length
                            
                            if 'public' in node.modifiers:
                                num_public_methods += 1

                
                    num_fields = 0

                    for _, node in tree.filter(javalang.tree.FieldDeclaration):
                        if isinstance(node, javalang.tree.FieldDeclaration):
                            num_fields += 1
                    
                
                    num_method_invocations = 0

                    for path, node in tree:
                        if isinstance(node, javalang.tree.MethodInvocation):
                            num_method_invocations += 1
                    
                    num_statements = 0

                    for path, node in tree:
                        if isinstance(node, javalang.tree.Statement):
                            num_statements += 1

                        #This line of code was made to remove block statements. 
                        if isinstance(node, javalang.tree.BlockStatement):
                            num_statements -= 1


        
                    num_interfaces=0 

                    for _, node in tree.filter(javalang.tree.ClassDeclaration):

                        if node.implements is not None: 
                         num_interfaces = len(node.implements)
                        
                    num_block_comments = 0
                    num_words =0

                    for _, node in tree.filter(javalang.tree.Documented):

                        if node.documentation is not None:
                            num_block_comments +=1
                            num_words += len(re.findall('\\w+', node.documentation))
                    
                    avg_method_name_length = 0 

                    if num_methods !=0: 

                        avg_method_name_length = sum_of_letters_method_names / num_methods
                        

                    max_num_statements = 0
                    max_num_conditional_loops = 0
                    max_num_exceptions = 0
                    max_num_return_points = 0
                    num_block_statements =0
                    

                    for _, node in tree.filter(javalang.tree.MethodDeclaration):

                        num_statements_method = 0
                        num_conditional_loops = 0
                        num_exceptions = 0
                        num_return_points = 0

                        for _, child_node in node.filter(javalang.tree.Statement):
            
                            num_statements_method += 1
                            if isinstance(child_node, (javalang.tree.IfStatement, javalang.tree.ForStatement,
                                                    javalang.tree.WhileStatement, javalang.tree.DoStatement,
                                                    javalang.tree.SwitchStatement)):
                                num_conditional_loops += 1

                            #Remove Block statements 
                            if isinstance(child_node, javalang.tree.BlockStatement):
                                num_block_statements += 1

                        for _, child_node in node.filter(javalang.tree.ThrowStatement):
                            num_exceptions += 1

                        for _, child_node in node.filter(javalang.tree.ReturnStatement):
                            num_return_points += 1
                        
                        num_statements_method_clean = num_statements_method-num_block_statements
                        
                        max_num_statements = max(max_num_statements, num_statements_method_clean)
                        max_num_conditional_loops = max(max_num_conditional_loops, num_conditional_loops)
                        max_num_exceptions = max(max_num_exceptions, num_exceptions)
                        max_num_return_points = max(max_num_return_points, num_return_points)
                    
                    dcm =0

                    if num_statements !=0:
                        dcm = num_words/num_statements

            
                    class_info.append({

                        "class_name": class_name, #name of the class
                        "MTH": num_methods, #Number of methods
                        "FLD": num_fields, #Number of Fields
                        "RFC": num_public_methods + num_method_invocations, #sum of public methods + sum method invocations
                        "INT": num_interfaces, #number of implemented interfaces

                        "SZ": max_num_statements, #MAX statements 
                        "CPX": max_num_conditional_loops, #MAX conditional + loop statements 
                        "EX": max_num_exceptions, #MAX exceptions in throw clause 
                        "RET": max_num_return_points, #MAX return points

                        "BCM": num_block_comments, #Blocks of Comments 
                        "NML": avg_method_name_length, #average length of methods names. 
                        "WRD": num_words, #count of words in comments blocks
                        "DCM": dcm, #total number of words in comments/total number of statements  
                    })

  
    df = pd.DataFrame(class_info)

    df.to_csv(r"feature_vectors/feature_vectors_not_labeled.csv")

    filtered_df2 = df.loc[df['class_name']=='PeepholeSimplifyRegExp']
    filtered_df3 = df.loc[df["class_name"]=="MinimizeExitPoints"]

    print(filtered_df2)
    print(filtered_df3)


if __name__=="__main__":

 folder_name = r"feature_vectors"
 extract_features(folder_name)
 df_analysis = pd.read_csv(r'feature_vectors/feature_vectors_labeled.csv')

 for column in df_analysis:
    
    count_buggy = (df_analysis['buggy'] == 1).sum()
    total_count = df_analysis.count()
