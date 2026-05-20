import numpy as np

class Node:
    def __init__(self, feature = None, left = None, right = None, value = None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value
        
class ManualDecisionTree:
    def __init__(self, max_depth=10, min_sample_split=5):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

        self.tree = None
        
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        
        for c in counts:
            p = c / len(y)
            impurity -= p ** 2
        return impurity

    def split(self, X, y, feature):
        left = X[:, feature] == 0
        right = X[:, feature] == 1
        
        return (X[left], y[left], X[right], y[right])

    def split_gini(self, X, y, feature):
        X_left, y_left, X_right, y_right = self.split(X, y, feature)
        n = len(y)
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 999
        
        g_left = self.gini(y_left)
        g_right = self.gini(y_right)
        
        weighted = len(y_left)/n * g_left + len(y_right) / n * g_right
        
        return weighted

    def best_split(self, X, y):
        best_feature = None
        best_gini = 999
        
        n_features = X.shape[1]
        
        for f in range(n_features):
            g = self.split_gini(X, y, f)
            
            if g < best_gini:
                best_gini = g
                best_feature = f
        return best_feature
    
    def build_tree(self, X, y, depth, max_depth):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        
        if depth >= max_depth:
            maj = np.bincount(y).argmax()
            return Node(value=maj)
        
        if len(y) < self.min_sample_split:
            majority = np.bincount(y).argmax()
            return Node(value=majority)
        
        feature = self.best_split(X,y)
        X_left, y_left, X_right, y_right = self.split(X,y,feature)
        
        if len(y_left) == 0 or len(y_right) == 0:
            maj = np.bincount(y).argmax()
            return Node(value=maj)
        
        l_child = self.build_tree(X_left, y_left, depth= depth + 1, max_depth= max_depth)
        r_child = self.build_tree(X_right, y_right, depth= depth + 1, max_depth= max_depth)
        
        return Node(feature=feature, left=l_child, right=r_child)
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, 0, self.max_depth)

    def predict_one(self, node, x):
        if node.value is not None:
            return node.value
        
        if x[node.feature] == 0:
            return self.predict_one(node.left, x)
        else:
            return self.predict_one(node.right, x)
        
    def predict(self, X):
        predictions = []
        
        for x in X:
            pred = self.predict_one(self.tree, x)
            predictions.append(pred)
        return np.array(predictions)