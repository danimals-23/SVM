import numpy as np
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """
    return np.dot(X1, X2.T)


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # Compute the Euclidean distance between the input vectors
    distance = np.linalg.norm(X1 - X2)

    # Compute the value of the Gaussian kernel function
    kernel_value = np.exp(-distance**2 / (2 * sigma**2))

    # Return the kernel value
    return kernel_value


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """

    n_samples = X.shape[0]

    K = kernel(X, X)

    term1 = np.sum(a)

    term2 = 0
    for i in range(n_samples):
        for j in range(n_samples):
            term2 += a[i] * a[j] * y[i] * y[j] * K[i,j]

    # Combine the two terms
    obj = term1 - 0.5 * term2

    return obj

class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute 
          the kernel matrix from data matrices; that matrix should be an array 
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features 
          is the number of features. For kernel=”precomputed”, the expected shape 
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        # save alpha parameters, weights, and bias weight
        n_samples, n_features = X.shape
        self.a = np.zeros(y.shape)
        self.w = None
        self.b = None

        
        # TODO: Define the constraints for the optimization problem
        def alpha_constraint(alpha):
            return np.dot(alpha, y)

        constraints = [{'type': 'eq', 'fun': lambda a: np.dot(a, y)},
                       {'type': 'ineq', 'fun': lambda a: a}]
        
        # TODO: Use minimize from scipy.optimize to find the optimal Lagrange multipliers
        result = minimize(lambda a: -objective_function(X, y, a, self.kernel), self.a, constraints=constraints)

        
        self.a = np.array(result.x)
        support_vectors = np.array(range(y.shape[0]))[self.a>1e-8]
        print(support_vectors)

        # TODO: Substitute into dual problem to find weights
        self.w = np.dot(self.a * y, X)

        # TODO: Substitute into a support vector to find bias
        
        self.support_vectors_ = X[support_vectors]
        self.b = (y[support_vectors] - np.dot(self.w, self.support_vectors_.T)).mean()


        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement

        decision = np.dot(X, self.w) + self.b

        # Make predictions
        y_pred = np.sign(decision)
        
        return y_pred

    def outputs(X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement

        return None


# from svm import SVM, linear_kernel
from sklearn.svm import SVC

class OvRSVM:
    def __init__(self, kernel='linear'):
        self.models = []
        self.kernel = kernel

    def fit(self, X, y):
        # create a binary classifier for each class
        classes = np.unique(y)
        for c in classes:
            y_binary = np.zeros_like(y)
            y_binary[y == c] = 1
            model = SVC(kernel=self.kernel, probability=True)
            model.fit(X, y_binary)
            self.models.append((c, model))

    def predict_prob(self, X):
        # compute probability estimates for each binary classifier
        probas = []
        for c, model in self.models:
            proba = model.predict_proba(X)[:, 1]
            probas.append(proba)
        # merge probability estimates into a single array
        probas = np.array(probas).T
        return probas

    def predict(self, X):
        # predict the class with the highest probability estimate
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


from sklearn.linear_model import LogisticRegression
class OvR_LogisticRegression():
    def __init__(self, solver='lbfgs', max_iter=100):
        self.solver = solver
        self.max_iter = max_iter
        self.classifiers = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for i, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1, 0)
            clf = LogisticRegression(solver=self.solver, max_iter=self.max_iter)
            clf.fit(X, y_binary)
            self.classifiers.append((cls, clf))

    def predict_prob(self, X):
        predictions = []
        for clf in self.classifiers:
            cls, binary_clf = clf
            y_pred = binary_clf.predict_proba(X)[:, 1]
            predictions.append(y_pred)
        return np.vstack(predictions).T
