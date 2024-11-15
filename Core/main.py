# Mehran Nosrati | 1403 | ML

import os
import time
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

current_dir = os.path.dirname(__file__)
type = os.path.join(current_dir, 'training_attack_types.txt')
name = os.path.join(current_dir, 'kddcup.names')
file = os.path.join(current_dir, 'kddcup.data.gz')

drop = [
    'num_root', 'srv_serror_rate', 'srv_rerror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'dst_host_same_srv_rate'
]

class PD:
    def __init__(self, drop, path, type, name, file):
        self.path = path
        self.drop = drop
        self.type = type
        self.name = name
        self.file = file
    
    def phg(self, df, features):
        plt.figure(figsize=(15, 6))
        df[features].hist(bins=20, color='#461c48', figsize=(15, 6))
        plt.suptitle('Distribution of Features', fontsize=16, fontweight='bold', color='#333333')
        plt.xlabel('Value', fontsize=14, fontweight='bold', color='#333333')
        plt.ylabel('Count', fontsize=14, fontweight='bold', color='#333333')
        for ax in plt.gcf().axes:
            ax.grid(True, linestyle='--', alpha=0.3, color='grey')
        plt.tight_layout()
        plt.savefig(f'{self.path}ALL_distribution.png', format='png', dpi=300)
        plt.close()
    
    def psh(self, feature, df):
        protocol_counts = df[feature].value_counts()
        colors=['#461c48', '#cb1b4e', '#f05d42', '#f58b63', '#f6b894', '#fae9d9']
        plt.figure(figsize=(15, 6))
        plt.bar(protocol_counts.index, protocol_counts.values, color=colors)
        plt.title(f'Distribution of {feature}', fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.path}{feature}_distribution.png', format='png', dpi=300)
        plt.close()
    
    def chm(self, df):
        corr = df.loc[:, df.dtypes == 'float64'].corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, 
                    xticklabels=corr.columns, 
                    yticklabels=corr.columns,
                    annot=True,
                    fmt='.2f',
                    linewidths=0.2,
                    cbar_kws={"shrink": .8},
                    square=True)
        plt.title('Correlation Heatmap', fontsize=18)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.savefig(f'{self.path}hm_distribution.png', format='png', dpi=300)
        plt.close()
    
    def numeric_features(self, df, filename):
        with open(self.path+filename, "w") as file:
            file.write("Numeric Features:\n")
            for column in df.select_dtypes(include=['number']).columns:
                file.write(f"Feature: {column}\n")
                file.write(f"Count: {df[column].count()}\n")
                file.write(f"Min Value: {df[column].min()}\n")
                file.write(f"Max Value: {df[column].max()}\n")
                file.write(f"Median: {df[column].median()}\n")
                file.write("-" * 30 + "\n")

    def categorical_features(self, df, filename):
        with open(self.path+filename, "w") as file:
            file.write("Categorical Features:\n")
            for column in df.select_dtypes(include=['object']).columns:
                file.write(f"Feature: {column}\n")
                file.write(f"Total Count: {df[column].count()}\n")
                file.write("Value Counts for Each Unique Value:\n")
                file.write(df[column].value_counts().to_string())
                file.write("\n" + "-" * 30 + "\n")
                
    def col(self):
        attack_types = {'normal': 'normal'}
        col_names = []
        
        with open(self.name, 'r') as fh:
            fh.readline()
            lines = fh.readlines()
        for line in lines:
            name = line.split(':')[0]
            col_names.append(name)
        col_names.append('Attack')

        with open(self.type, 'r') as fh:
            lines = fh.readlines()
        for line in lines:
            line = line.replace('\n', '')
            if not line: continue
            typ = line.split(' ')[0]
            cls = line.split(' ')[1]
            attack_types[typ] = cls
        
        return attack_types, col_names
    
    def red(self):
        A,C = self.col()
        df = pd.read_csv(self.file,names=C)
        df['Label'] = df.Attack.apply(lambda r:A[r[:-1]])
        df.drop('Attack',axis = 1,inplace = True)
        return df

    def ppd(self):
        df = self.red()
        self.psh('protocol_type',df)
        self.psh('service',df)
        self.psh('flag',df)
        self.psh('Label',df)
        self.numeric_features(df,"numeric_features.txt")
        self.categorical_features(df,"categorical_features.txt")
        le = LabelEncoder()
        for c in ['protocol_type', 'service', 'flag']:
            df[c] = le.fit_transform(df[c])
        scaler = MinMaxScaler()
        df[['protocol_type', 'service', 'flag']] = scaler.fit_transform(df[['protocol_type', 'service', 'flag']])
        
        self.phg(df, ['protocol_type', 'service', 'flag'])

        self.chm(df)
        df.drop(columns=self.drop, axis=1, inplace=True)
        return df
    
    def start(self):
        return self.ppd()

class ML:
    def __init__(self, path, model, test_size=0.3, random_state=42, pathrules="rules.txt", pathtree="tree.svg"):
        self.path = path
        self.pathrules = path+pathrules
        self.pathtree = path+pathtree
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()

    def log_operation(self, operation_name):
        with open(self.path + "operations_log.txt", "a") as f:
            f.write(f"{operation_name} executed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
    def model_details(self):
        print("Model Parameters:")
        for param, value in self.model.get_params().items():
            print(f"{param}: {value}")
         
    def load(self, df, label_col='Label'):
        X = df.drop(label_col, axis=1)
        Y = df[label_col]
        self.feature_names = X.columns.tolist()
        X = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
        self.log_operation("Load")

    def train(self):
        self.model_details()
        start_time = time.time()
        self.model.fit(self.X_train, self.Y_train)
        end_time = time.time()
        print(f'Time T: {end_time - start_time} seconds')
        self.log_operation("Train")

    def evaluate(self):
        start_time = time.time()
        Y_test_pred = self.model.predict(self.X_test)
        end_time = time.time()
        print(f'Time E: {end_time - start_time} seconds')
        accuracy = accuracy_score(self.Y_test, Y_test_pred)
        precision = precision_score(self.Y_test, Y_test_pred, average='weighted')
        recall = recall_score(self.Y_test, Y_test_pred, average='weighted')
        f1 = f1_score(self.Y_test, Y_test_pred, average='weighted')
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self.log_operation("Evaluate")
        return results
        
    def rules(self):
        all_rules = ""
        if hasattr(self.model, 'estimators_'):
            for i, tree in enumerate(self.model.estimators_[:5]):
                if isinstance(self.model, GradientBoostingClassifier):
                    tree = tree[0]
                rules = export_text(tree, feature_names=self.feature_names)
                all_rules += f"Rules for Tree {i + 1}:\n{rules}\n"
                all_rules += "-" * 50 + "\n"
        else:
            rules = export_text(self.model, feature_names=self.feature_names)
            all_rules += f"Rules for the Decision Tree:\n{rules}\n"
        with open(self.pathrules, "w") as file:
            file.write(all_rules)
            file.write("\nTotal number of rules: " + str(all_rules.count('\n')))
        print(f"Rules have been saved to {self.pathrules}")
        self.log_operation("Rules")

    def Tree(self):
        if hasattr(self.model, 'estimators_'):
            for i, tree in enumerate(self.model.estimators_[:5]):
                if isinstance(self.model, GradientBoostingClassifier):
                    tree = tree[0]
                plt.figure(figsize=(400, 100))
                plot_tree(tree, feature_names=self.feature_names, filled=True, fontsize=10)
                plt.savefig(f"{self.pathtree}{i + 1}.svg", format="svg")
                plt.close()
                print(f"Tree {i + 1} has been saved to {self.pathtree}{i + 1}.svg")
        else:
            plt.figure(figsize=(400, 100))
            plot_tree(self.model, feature_names=self.feature_names, filled=True, fontsize=10)
            plt.savefig(self.pathtree, format="svg")
            plt.close()
            print(f"Tree has been saved to {self.pathtree}")
            self.log_operation("Tree")
    
    def lgb(self, M):
        fig, ax = plt.subplots(nrows=2, figsize=(50,10), sharex=True)
        lgb.plot_tree(M, tree_index=0,dpi=10, ax=ax[0])
        lgb.plot_tree(M, tree_index=1,dpi=10, ax=ax[1])
        plt.savefig(self.pathtree, format="jpg", bbox_inches='tight')
        plt.close()
        print(f"lgb has been saved to {self.pathtree}")
        self.log_operation("Lgb")
        
    def result(self, R):
        print(f'Accuracy: {R["accuracy"]}')
        print(f'Precision: {R["precision"]}')
        print(f'Recall: {R["recall"]}')
        print(f'F1 Score: {R["f1"]}')

def main():
    PData = PD(drop, "temp/", type, name, file)
    df = PData.start()

    Name = "Decision Tree Classifier"
    print(f'N: {Name}')
    DT = DecisionTreeClassifier(criterion="entropy", max_depth = 17, random_state=7)
    detector = ML("temp/result/" ,DT, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[tree].svg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.rules()
    detector.Tree()
    
    Name = "Random Forest Classifier"
    print(f'N: {Name}')
    RF = RandomForestClassifier(n_estimators=100, max_depth = 17, random_state=7)
    detector = ML("temp/result/" ,RF, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[tree].svg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.rules()
    detector.Tree()
    
    Name = "Gradient Boosting Classifier"
    print(f'N: {Name}')
    GB = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=7)
    detector = ML("temp/result/" ,GB, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[tree].svg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.rules()
    detector.Tree()
    
    Name = "ExtraTrees Classifier"
    print(f'N: {Name}')
    ET = ExtraTreesClassifier(n_estimators=20, random_state=7)
    detector = ML("temp/result/" ,ET, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[tree].svg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.rules()
    detector.Tree()
    
    Name = "LGBM Classifier"
    print(f'N: {Name}')
    LGBM = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=7)
    detector = ML("temp/result/" ,LGBM, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[LGBM].jpg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.lgb(LGBM)
    
    Name = "AdaBoost Classifier"
    print(f'N: {Name}')
    AB = AdaBoostClassifier(DT, algorithm='SAMME', n_estimators=20, random_state=7)
    detector = ML("temp/result/" ,AB, pathrules=f"{Name}[rules].txt", pathtree=f"{Name}[tree].svg")
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    detector.rules()
    detector.Tree()
    
    Name = "Naive Bayes"
    print(f'N: {Name}')
    NB = GaussianNB(var_smoothing=1e-9)
    detector = ML("temp/result/" ,NB)
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    
    Name = "K-Nearest Neighbors"
    print(f'N: {Name}')
    KNN = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=500)
    detector = ML("temp/result/" ,KNN)
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)

    Name = "Support Vector Machine"
    print(f'N: {Name}')
    SVM = LinearSVC(random_state=7)
    detector = ML("temp/result/" ,SVM)
    detector.load(df)
    detector.train()
    R = detector.evaluate()
    detector.result(R)
    
if __name__ == '__main__':
    main()