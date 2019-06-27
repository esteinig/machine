import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import itertools
# import seaborn as sns
import random

# random.seed(1)


class DataGenerator:
    """
    A class used to generate classic random datasets for machine learning classification

    (see e.g. https://playground.tensorflow.org)
    ...

    Attributes
    ----------
    n_samples : int, optional
        number of random points generated for each class in n_targets (default is 100)
    targets : int, optional
        number of distinct target classes of categories (default is 2)
    n_features : int, optional (default is 2)
        dimension of dataset, i.e. the number of independent explanatory variables
    noise : float, optional (default is 0.0)
        level of absolute random noise added to each base (deterministic) dataset

    Methods
    -------
    make_cloud(centres=None, n_samples=100, targets=2, noise=None, dim=2)
        Generates Gaussian clouds of dimension dim
    make_donut(radii=None, n_samples=100, targets=2, noise=None)
        Generatea a dataset of noisy concentric rings
    make_xor(n_samples=100, noise=None, dim=2)
        Generates a [-1,1]^d hypercube of uniformly distributed points with target = sign(x1*x2*...*xn)
    make_spiral(n_samples=100, targets=2, noise=None)
        Generates a set of uniformly spaced spiral arms
    make_moon(n_samples=100)
        Make two interleaving half circles
    """

    def __init__(self):
        return

    def _random_radius(self, radius, n_samples, noise):
        """
        Generates a normally distributed random list centred on radius

        ...

        Parameters
        ----------
        radius: np.array
            Array of radii
        n_samples: int
            Number of random points to be generated
        noise: float
            scale parameter for the normally distributed data

        Returns
        -------
            float
            normally distributed list with mean = radius, scale = noise_level, size = n_points
        """
        return np.random.normal(loc=radius, scale=noise, size=n_samples)

    def _distribute_samples(self, n_samples, n_targets):
        """
        Distributes samples evenly among output classes

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (spiral arms)

        Returns
        -------
        target_samples : lst
            List of sample distributions
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """
        target_samples = [n_samples // n_targets] * n_targets
        for rem in range(n_samples % n_targets):
            target_samples[rem] += 1
        target_samples = [int(i) for i in target_samples]
        y = np.array([i for lst in [[t] * target_samples[t] for t in range(n_targets)] for i in lst]).reshape((n_samples,))
        return target_samples, y

    def make_donut(self, radii=None, n_samples=100, n_targets=2, noise=0.0):
        """
        Generates concentric doughnut rings

        A simple toy dataset to visualize clustering and classification algorithms.

        Parameters
        ----------
        radii : float, optional (default=None, populated as sequential integers at start of code)
            The radii of the donut rings
        n_samples : int, optional (default=100)
            The total number of points to be generated.
        n_targets : int, optional (default=2)
            The number of target classes (doughnuts)
        noise : double (default=0.0)
            Standard deviation of Gaussian noise added to the radius.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """

        # default option to just use sequential integer values for the radii of the doughnuts
        if not radii:
            radii = list(range(n_targets))

        # otherwise check that the right number of radii have been requested
        elif n_targets != len(radii):
            raise ValueError("provided radii not equal to number of targets requested")

        # work out how many samples in each group are needed, set the y-values, initialise empty list for x-values
        target_samples, y = self._distribute_samples(n_samples, n_targets)
        X = []

        # loop through targets
        for t in range(n_targets):
            sample_size = target_samples[t]
            r = radii[t]

            # find the angle
            theta = np.random.uniform(0, 2 * np.pi, sample_size)

            # find the coordinates from the angle and the radius - with noise applied to the radius, if any
            X0 = self._random_radius(r, sample_size, noise) * np.cos(theta)
            X1 = self._random_radius(r, sample_size, noise) * np.sin(theta)
            X.append(np.vstack((X0, X1)).T)

        # reorganise output structure
        X = np.concatenate(X, axis=0)
        return X, y

    def make_cloud(self, centres=None, n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None):
        """
        Generates Gaussian clouds of dimension n_features

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (spiral arms)
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        n_features : int, optional (default=2)
            Dimension of each cloud
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """

        if centres is None:
            mean = [0] * n_features
            cov = 2*np.identity(n_features)
            centres = np.random.multivariate_normal(mean, cov, n_targets)
        else:
            n_features = len(centres[0])
            n_targets = len(centres)
            centres = np.array(centres).reshape(n_targets, n_features)

        target_samples, y = self._distribute_samples(n_samples, n_targets)
        cov = noise*np.identity(n_features)
        X = []
        for t in range(n_targets):
            sample_size = target_samples[t]
            X.append(np.random.multivariate_normal(centres[t, :], cov, sample_size))
        X = np.concatenate(X, axis=0)

        data = X, y
        return data

    def make_spiral(self, n_samples=100, n_targets=2, noise=0.05, random_state=None, inner_radius=0.0, outer_radius=2):
        """Make a set of uniformly spaced spiral arms

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (spiral arms)
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., n_targets) for class membership of each sample.
        """

        target_samples, y = self._distribute_samples(n_samples, n_targets)

        X = []
        for t in range(n_targets):
            sample_size = target_samples[t]
            theta = np.sqrt(np.linspace(0, 16*np.pi**2, sample_size)) + 2 * t * np.pi / n_targets
            r = np.linspace(inner_radius, outer_radius, sample_size)
            random_r = self._random_radius(radius=r, n_samples=sample_size, noise=noise)
            X.append(np.vstack([random_r * np.cos(theta), random_r*np.sin(theta)]).T)
        X = np.concatenate(X, axis=0)
        data = X, y
        return data

    def make_xor(self, n_samples=100, noise=None, n_features=2, n_targets=2, random_state=None):
        """Make a hypercube of of uniformly distributed points where y = sign(x0*x1*...)

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        n_features : int, optional (default=2)
            Dimension of xor hypercube
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0 or 1) for class membership of each sample.
        """

        X = np.random.uniform(low=-1, high=1, size=(n_samples, n_features))
        y = (((np.sign(np.prod(X, axis=1)) + 1) / 2).astype(int)).reshape(n_samples,)

        if noise is not None:
            X += np.random.normal(scale=noise, size=X.shape)
        data = X, y
        return data

    def make_moons(self, n_samples=100, noise=None, n_targets=2):
        """Make two interleaving half circles

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        X : array of shape [n_samples, 2]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0 or 1) for class membership of each sample.
        """

        n_targets = 2

        target_samples, y = self._distribute_samples(n_samples, n_targets)
        outer_circ_x = np.cos(np.linspace(0, np.pi, target_samples[0]))
        outer_circ_y = np.sin(np.linspace(0, np.pi, target_samples[0]))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, target_samples[1]))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, target_samples[1])) - .5

        X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y)]).T
        if noise is not None:
            X += np.random.normal(scale=noise, size=X.shape)
        data = X, y
        return data


class DataContainer:
    """
        A class used to store and manipulate datasets for machine learning classification

        ...

        Attributes
        ----------
        data_raw : array
            raw (e.g. unscaled, unshuffled) dataset given as a tuple (X,y) of np.arrays
        data_df : dataframe
            dataset saved as a a pandas dataframe
        n_samples : int
            number of samples/observations in the dataset
        n_features : int
            dimension of dataset, i.e. the number of independent explanatory variables
        feature_names : char
            list of feature names, e.g. X0, X1, ...
        n_targets : int
            no. of distinct target labels (for categorical data)

        Methods
        -------
        shuffle()
            Randomly shuffles the dataset and updates the attribute self.data_df
        train_test_split(frac=0.8)
            Randomly split the data into train and test sets
        plot()
            Plot the dataset
        scale()
            Scale the input data X according to the mean and standard deviation of each feature
        back_transform()
            Invert the normal scaling applied by the scale() method
        add_polynomial_features(degree=2, terms=None, powers_only=False, interaction_only=False, inplace=True)
            Add power and interaction features up to degree=degree to the original dataset, e.g. X0**2, X0*X1
        add_white_noise()
            Add a new feature that is purely white noise
        """

    def __init__(self, data, feature_names=None,
                 target_names=None, shuffle_data=True):
        self.data_raw = data
        self.feature_names = feature_names
        self.target_names = target_names
        self.data_df = self._store_data_as_df(data, feature_names, target_names)
        self.n_features = len(self.feature_names)
        self.n_targets = len(self.target_names)
        self.n_samples = self.data_df.shape[0]
        if shuffle_data:
            self.shuffle()
        else:
            self.shuffled = False
        self.scales = [np.ones(shape=(self. n_features, 1)), np.zeros(shape=(self.n_features, 1))]

    def _store_data_as_df(self, data, feature_names, target_names):
        """
        Store data array as a dataframe

        Parameters
        ----------
        data : array, shape = (n_samples, n_features + 1)
            Original dataset stored as a numpy array
        Returns
        data : dataframe, shape = (n_samples, n_features + 1)
            Original dataset stored as a pandas dataframe
        """
        X, y = data
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [''.join(['X', str(i)]) for i in range(n_features)]
        if target_names is None:
            if y.ndim == 1:
                target_names = ['y']
            else:
                target_names = [''.join(['y', str(i)]) for i in range(y.shape[1])]
        n_targets = len(target_names)
        colnames = feature_names + target_names
        data = pd.DataFrame(np.concatenate((X, y.reshape(n_samples, n_targets)), axis=1), columns=colnames)
        self.feature_names = feature_names
        self.target_names = target_names
        return data

    def _extract_arrays(self, data_df=None):
        if data_df is None:
            data_df = self.data_df
        y = np.array(data_df[self.target_names]).flatten()
        X = np.array(data_df[self.feature_names])
        return X, y

    def shuffle(self):
        """Shuffle dataset stored in self.data_df

        """
        self.data_df = self.data_df.sample(frac=1)
        self.shuffled = True

    def train_test_split(self, frac=0.8):
        """Split the dataset self.data_df into training and test sets


        Parameters
        ----------
        frac : float, optional (default=0.8)
            Fraction of the dataset assigned to the training set (with the remaining
            (1 - frac) assigned to the test set

        Returns
        -------
        X_train : array of shape [frac*n_samples, n_features]
            The training samples.
        y_train : array of shape [n_samples, 1]
            The training targets.
        X_test : array of shape [(1 - frac)*n_samples, n_features]
            The test samples.
        y_test : array of shape [n_samples, 1]
            The test targets
        """

        if not self.shuffled:
            self.shuffle()
        train = self.data_df.sample(frac=frac)
        test = self.data_df.loc[~self.data_df.index.isin(train.index), :]

        X_train, y_train = self._extract_arrays(train)
        X_test, y_test = self._extract_arrays(test)

        return X_train, y_train, X_test, y_test

    def plot(self):
        """
        Plot the dataset

        """

        X, y = self._extract_arrays()
        if self.n_features == 2:
            plt.scatter(X[:, 0], X[:, 1], c=y,
                        edgecolors='w', linewidths=0.5)
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])
        elif self.n_features == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
                       edgecolors='w', linewidths=0.5)
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            ax.set_zlabel(self.feature_names[2])
        plt.grid(linewidth=0.2)
        plt.show()
        return

    def scale(self, X=None, inplace=True):
        """
        Scale the input to have mean=0 and sd=1 for each feature

        Parameters
        ----------
        X : array (optional)
            Optional array to scale. If None (default) self.data_df is scaled
        inplace : Boolean (optional, default=True)
            Boolean argument to determine whether the scaled array should replace the original

        Returns
        -------
        X_scaled : array
            Scaled array where each feature has mean=0 and sd=1
        """
        if X is None:
            X, _ = self._extract_arrays()
        _, y = self._extract_arrays()
        Xbar = np.mean(X, axis=0)
        Xsig = np.std(X, axis=0)
        self.scales = [Xbar, Xsig]
        X_scaled = (X - Xbar)/Xsig
        if inplace:
            data = X_scaled, y
            self.data_df = self._store_data_as_df(data, self.feature_names, self.target_names)
        return X_scaled

    def back_transform(self, X_scaled=None, scales=None, inplace=True):
        """
        Back transform the scaled array to the untransformed original

        Parameters
        ----------


        Returns
        -------
        """
        _, y = self._extract_arrays()
        if X_scaled is None:
            X_scaled, _ = self._extract_arrays()
            scales = self.scales
        Xbar, Xsig = scales
        X = X_scaled*Xsig + Xbar
        if inplace:
            data = X, y
            self.data_df = self._store_data_as_df(data, self.feature_names, self.target_names)
        return X

    def add_polynomial_features(self, degree=2, terms=None, powers_only=False,
                                interaction_only=False, inplace=True):
        """
        Transform the original dataset X by adding polynomial and interaction features, e.g. X0**2, X1*X2

        Parameters
        ----------
        degree : int (default=2)
            The degree of polynomial and interaction terms to be added to the original dataset. For example, if the
            original dataset had three features: X0; X1; and X2 and we set the degree = 2, the new dataset would contain
            the additional features: X0**2, X1**2, and X2**2 and X0*X1, X0*X2 and X1*X2.
        terms : list (default = None)
            A list of the features to be transformed, i.e. if only a subset of features are of interest
        powers_only: Boolean (default = False)
            Include only polynomial terms, e.g. X0**3
        interaction_only: Boolean (default = False)
            Include only interaction terms, e.g. X0*X1
        in_place : Boolean (default = True)
            Store the modified (i.e. extended) array in place of the original dataset

        Returns
        -------
        X_new : array
            Extended array containing additional polynomial and interaction features
        """
        if terms is None:
            X_subset = self.data_df[self.feature_names]
        else:
            X_subset = self.data_df[terms]
        colnames = list(X_subset)
        X_new = pd.DataFrame()
        new_feature_names = []
        if not powers_only:
            for deg in range(2, degree + 1):
                for combination in itertools.combinations(colnames, deg):
                    prod = X_subset[combination[0]]
                    for feature_name in combination[1:]:
                        prod = prod*X_subset[feature_name]
                    new_feature_name = '.'.join(combination)
                    new_feature_names.append(new_feature_name)
                    X_new[new_feature_name] = prod
        if not interaction_only:
            for deg in range(2, degree + 1):
                for feature in colnames:
                    new_feature_name = feature + '**' + str(deg)
                    new_feature_names.append(new_feature_name)
                    X_new[new_feature_name] = X_subset[feature]**deg
        new_feature_names = list(X_new)
        if inplace:
            feature_names = self.feature_names + new_feature_names
            X_new = np.asarray(X_new[new_feature_names])
            X, y = self._extract_arrays()
            data = np.concatenate((X, X_new), axis=1), y
            self.data_df = self._store_data_as_df(data, feature_names, self.target_names)
        return X_new

    def add_features(self, X_new, feature_names=None):
        n_new_features = X_new.shape[1]
        X, y = self._extract_arrays()
        if feature_names is None:
            feature_names = self.feature_names + [''.join(['NF', str(f)]) for f in range(n_new_features)]
        else:
            feature_names = self.feature_names + feature_names
        data = np.concatenate((X, X_new), axis=1), y
        self.data_df = self._store_data_as_df(data, feature_names, self.target_names)
        return

    def one_hot_enc(self, y=None):
        """
        Convert one-dimensional target lists or arrays to one-hot encoded arrays

        Parameters
        ----------
        y : list/array (default = None)
            List/array of targets

        Returns
        -------
        y_one_hot : array, shape = (n_samples, n_targets)
            One-hot encoded target array
        """
        if y is None:
            X, y = self._extract_arrays()
        n_samples = y.shape[0]
        n_targets = len(np.unique(y))
        y_one_hot = np.zeros(shape=(n_samples, n_targets))
        for i in range(n_samples):
            col = int(y[i])
            y_one_hot[i, col] = 1
        return y_one_hot

    def add_white_noise(self):
        """
        Add a new feature of random white noise

        ...

        Returns
        -------
        X : array of shape [n_samples, n_features + 1]
            The generated samples plus a new column of white noise.
        """
        if self.data_df is None:
            raise NotImplementedError('Dataset not found (i.e. not yet generated)')
        n_samples = self.n_samples
        X, y = self._extract_arrays()
        white_noise = np.array(np.random.normal(size=n_samples)).reshape(n_samples, 1)
        X = np.hstack((X, white_noise))
        feature_names = self.feature_names + ['Noise']
        data = X, y
        self.data_df = self._store_data_as_df(data, feature_names, self.target_names)
        return

    def pca(self, X=None, plot=False):
        if X is None:
            X, _ = self._extract_arrays()
        covX = np.cov(X.T)
        lambdas, Q = np.linalg.eigh(covX)

        idx = np.argsort(-lambdas)
        lambdas = lambdas[idx]
        lambdas = np.maximum(lambdas, 0)
        Q = Q[:, idx]
        Z = X.dot(Q)
        if plot:
            plt.plot(np.cumsum(lambdas)/np.sum(lambdas))
            plt.show()
        return Z, lambdas

    def pairplot(self, data=None):
        if data is None:
            data = self.data_df
        n_vars = len(data.columns)
        var_names = list(data.columns)
        last = n_vars - 1
        target_name = var_names[-1]
        n_targets = len(np.unique(data[target_name]))
        for row in range(n_vars):
            for col in range(row):
                plt.subplot(n_vars, n_vars, row*n_vars + (col + 1))
                plt.scatter(data[var_names[col]], data[var_names[row]], c=data[target_name],
                            edgecolors='w', linewidths=0.5)
                plt.gca().axes.grid(linewidth=0.2)
                if col == 0:
                    plt.ylabel(var_names[row])
                if row == last:
                    plt.xlabel(var_names[col])
                if row != last:
                    plt.gca().xaxis.set_major_formatter(plt.NullFormatter())
                    # plt.gca().axes.get_xaxis().set_ticks([])
                if col != 0:
                    plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
                    # plt.gca().axes.get_yaxis().set_ticks([])
        covX = np.cov(data.T)
        # print('covX:\n', covX)
        for row in range(n_vars):
            for col in range(row+1, n_vars):
                plt.subplot(n_vars, n_vars, row*n_vars + (col + 1))
                plt.text(0.5, 0.5, s=str(np.round(covX[row, col], 4)), fontsize=1.5*np.log(np.abs(covX[row, col])) + 10,
                         horizontalalignment='center', verticalalignment='center')
                plt.gca().axes.get_xaxis().set_ticks([])
                plt.gca().axes.get_yaxis().set_ticks([])
        for idx in range(n_vars):
            plt.subplot(n_vars, n_vars, idx*(n_vars + 1) + 1)
            if idx == last:
                plt.hist(data[var_names[idx]])
            else:
                for target in range(n_targets):
                    data[data[target_name] == target][var_names[idx]].plot.kde(c=target)
                    plt.gca().fill()
            if idx == 0:
                plt.ylabel(var_names[idx])
            if idx == last:
                plt.xlabel(var_names[idx])
            if idx != 0:
                plt.gca().axes.get_yaxis().set_ticks([])
            if idx != last:
                plt.gca().axes.get_xaxis().set_ticks([])
        plt.show()


if __name__ == "__main__":
    generator = DataGenerator()
    doughnut = generator.make_donut()
    container = DataContainer(doughnut)
    container.plot()
