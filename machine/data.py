import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import itertools
import seaborn as sns


class DataGenerator:
    """
    A class used to generate classic random datasets for machine learning classification

    (see e.g. https://playground.tensorflow.org)
    ...

    Methods
    -------
    make_cloud(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None, centres=None)
        Generates Gaussian clouds of dimension dim
    make_donut(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None, radii=None)
        Generatea a dataset of noisy concentric rings
    make_xor(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None)
        Generates a [-1,1]^d hypercube of uniformly distributed points with target = sign(x1*x2*...*xn)
    make_spiral(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None)
        Generates a set of uniformly spaced spiral arms
    make_moons(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None)
        Make two interleaving half circles
    make_wheel(n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None)
        Generates a uniform pin wheel
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
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        target_samples : lst
            List of sample distributions
        indices:
            Locations of the values referring to each sample in the y array
        """

        # find the main y vector for the distribution of the data across targets
        targets = list(range(n_targets))
        repeats = n_samples // n_targets + 1
        y = targets * repeats
        y = np.sort(y[: n_samples])
        y = np.asarray(y).reshape(n_samples, ).astype(int)

        # find additional summary quantities to use later
        indices = [list(np.where(y == t)[0]) for t in targets]
        target_samples = [len(indices[t]) for t in targets]
        targets = list(range(n_targets))

        return y, target_samples, indices, targets

    def make_donut(self, n_samples=100, n_targets=2, noise: float = 0.1, radii=None):
        """
        Generates concentric doughnut rings

        A simple toy dataset to visualize clustering and classification algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (spiral arms)
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        radii : float, optional (default=None)
            The radii of the donut rings

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """

        # default option to just use sequential integer values from one for the radii of the doughnuts
        if not radii:
            radii = list(range(1, n_targets + 1))

        # otherwise check that the right number of radii have been requested
        elif n_targets != len(radii):
            raise ValueError("provided radii not equal to number of targets requested")

        # work out how many samples in each group are needed, set the y-values, initialise empty array for x-values
        sample_coordinates, target_samples, indexes, targets = self._distribute_samples(n_samples, n_targets)
        sample_coordinates = np.empty(shape=(n_samples, 2))

        # loop through targets
        for t in targets:
            sample_size = target_samples[t]
            rows = indexes[t]

            # find the angle and the radius
            theta = np.random.uniform(0, 2 * np.pi, sample_size)
            r = self._random_radius(radii[t], sample_size, noise)

            # find the coordinates from the angle and the radius - with noise applied to the radius, if any
            sample_coordinates[rows, :] = np.vstack((r * np.cos(theta),
                                                     r * np.sin(theta))).T

        # return finished data
        return sample_coordinates, sample_coordinates

    def make_cloud(self, n_samples=100, n_targets=2, noise=0.05, n_features=2, random_state=None, centres=None):
        """
        Generates Gaussian clouds of dimension n_features

        A simple toy data set to visualize clustering and classification algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (clouds)
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        n_features : int, optional (default=2)
            Dimension of each cloud.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.
        centres : array of shape [n_targets, n_features], optional (default=None)
            Array of cloud centre points

        Returns
        -------
        sample_coordinates : array of shape [n_samples, n_features]
            The generated samples.
        sample_target_values : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """

        # for reproducibility, as described above
        if random_state:
            np.random.seed(random_state)

        # if centres provided as an array, then the features/dimensions is the width and the targets is the columns
        if centres:
            n_features, n_targets = len(centres[0]), len(centres)
            centres = np.array(centres).reshape(n_targets, n_features)

        # find centres of the requested clouds using a multivariate normal distribution if not provided
        else:
            centre_covariance = 2.0 * np.identity(n_features)
            centres = np.random.multivariate_normal([0.0] * n_features, centre_covariance, n_targets)

        # no correlation between sample positions across each feature/dimension
        sample_covariance = noise * np.identity(n_features)

        # standard initialisation
        sample_target_values, target_samples, indices, targets = self._distribute_samples(n_samples, n_targets)
        sample_coordinates = np.empty(shape=(n_samples, n_features))

        # generate cloud of samples for each target
        for n_target in targets:
            sample_size, rows = target_samples[n_target], indices[n_target]
            sample_coordinates[rows, :] = np.random.multivariate_normal(centres[n_target, :], sample_covariance, sample_size)
        return sample_coordinates, sample_target_values

    def make_spiral(self, n_samples=100, n_targets=2, noise=0.05, random_state=None, inner_radius=0.0, outer_radius=2):
        """Make a set of uniformly spaced spiral arms

        A simple toy dataset to visualize clustering and classification algorithms.

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
        inner_radius : float, optional (default=0.0)
            Inner radius of the spiral arm
        outer_radius : float, optional (default=2.0)
            Outer radius of the spiral arm

        Returns
        -------
        sample_coordinates : array of shape [n_samples, n_features]
            The generated samples.
        sample_target_values : array of shape [n_samples]
            The integer labels (0, 1, ..., n_targets) for class membership of each sample.
        """

        # for reproducibility, as described above
        if random_state:
            np.random.seed(random_state)

        # standard initialisation
        sample_target_values, target_samples, indices, targets = self._distribute_samples(n_samples, n_targets)
        sample_coordinates = np.empty(shape=(n_samples, 2))

        # generate spiral of samples for each target
        for n_target in targets:
            sample_size, rows = target_samples[n_target], indices[n_target]
            theta = np.sqrt(np.linspace(0, 16 * np.pi ** 2, sample_size)) + 2 * np.pi * n_target / n_targets
            random_radius = self._random_radius(radius=np.linspace(inner_radius, outer_radius, sample_size),
                                                n_samples=sample_size, noise=noise)
            sample_coordinates[rows, :] = np.vstack((random_radius * np.cos(theta), random_radius * np.sin(theta))).T
        return sample_coordinates, sample_target_values

    def make_xor(self, n_samples=100, noise=0.05, n_features=2, random_state=None):
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
        sample_coordinates : array of shape [n_samples, n_features]
            The generated samples.
        sample_target_values : array of shape [n_samples]
            The integer labels (0 or 1) for class membership of each sample.
        """

        # for reproducibility, as described above
        if random_state:
            np.random.seed(random_state)

        # generate xor data for samples
        sample_coordinates = np.random.uniform(low=-1, high=1, size=(n_samples, n_features))
        sample_coordinates += np.random.normal(scale=noise, size=sample_coordinates.shape)
        sample_target_values = ((np.sign(np.prod(sample_coordinates, axis=1)) + 1) / 2).astype(int)
        return sample_coordinates, sample_target_values

    def make_moons(self, n_samples=100, noise=0.1, random_state=None):
        """Make two interleaving half circles

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (intervleaving moons). Current implementation
            requires n_targets = 2.
        noise : double (default=0.1)
            Standard deviation of Gaussian noise added to the data.
        n_features : int, optional (default=2)
            Dimension of the moons. Current implementation requires n_features = 2.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        sample_coordinates : array of shape [n_samples, 2]
            The generated samples.
        sample_target_values : array of shape [n_samples]
            The integer labels (0 or 1) for class membership of each sample.
        """

        # for reproducibility, as described above
        if random_state:
            np.random.seed(random_state)

        # must have two targets and two features for this method
        n_targets = 2

        # standard initialisation
        sample_target_values, target_samples, indices, _ = self._distribute_samples(n_samples, n_targets)

        # central positions for upper half-moon
        outer_circ_x = np.cos(np.linspace(0, np.pi, target_samples[0]))
        outer_circ_y = np.sin(np.linspace(0, np.pi, target_samples[0]))

        # central positions for lower half-moon
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, target_samples[1]))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, target_samples[1])) - .5

        # bring together and add noise
        sample_coordinates = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
        if noise:
            sample_coordinates += np.random.normal(scale=noise, size=sample_coordinates.shape)
        return sample_coordinates, sample_target_values

    def make_wheel(self, n_samples=100, n_targets=2, noise=0.05, random_state=None):
        """Make pin wheel

        A simple toy dataset to visualize clustering and classification algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (pieces of the wheel).
        noise : double, optional (default=0.05)
            Standard deviation of Gaussian noise added to the data.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        sample_coordinates : array of shape [n_samples, 2]
            The generated samples.
        sample_target_values: array of shape [n_samples]
            The integer labels (0 or 1) for class membership of each sample.
        """

        # for reproducibility, as described above
        if random_state:
            np.random.seed(random_state)

        # standard initialisation
        sample_target_values, target_samples, indices, targets = self._distribute_samples(n_samples, n_targets)

        # generate data, must be with two features for this method
        sample_coordinates = np.empty(shape=(n_samples, 2))
        for n_target in targets:
            sample_size, rows = target_samples[n_target], indices[n_target]
            theta = np.random.uniform(low=2 * n_target * np.pi / n_targets, high=2 * (n_target + 1) * np.pi / n_targets,
                                      size=(sample_size,))
            central_radius = np.random.uniform(low=0, high=1, size=(sample_size,))
            sample_coordinates[rows, :] = np.vstack((central_radius * np.cos(theta),
                                                     central_radius * np.sin(theta))).T

        # add noise if requested and return
        if noise:
            sample_coordinates += np.random.normal(loc=0.0, scale=noise, size=sample_coordinates.shape)
        return sample_coordinates, sample_target_values


class DataContainer:
    """
        A class used to store and manipulate datasets for machine learning classification

        ...

        Attributes
        ----------
        data_raw : array
            raw (e.g. unscaled, unshuffled) dataset given as a tuple (X,y) of np.arrays
        data : dataframe
            dataset saved as a a pandas dataframe
        n_samples : int
            number of samples/observations in the dataset
        n_features : int
            dimension of dataset, i.e. the number of independent explanatory variables
        feature_names : char
            list of feature names, e.g. X0, X1, ...
        target_name : char
            name of target variable
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
                 target_name=None, shuffle_data=False):
        # print('Data is pd: %s' % isinstance(data, pd.DataFrame))
        if not isinstance(data, pd.DataFrame):
            self.data_raw = data
            self.data = self._store_data_as_df(data, feature_names, target_name)
        else:
            self.data = data
            if feature_names is None:
                self.feature_names = list(self.data)[:-1]
            if target_name is None:
                self.target_name = list(self.data)[-1]
        self.n_features = len(self.feature_names)
        # self.n_targets = len(self.target_name)
        self.n_samples = self.data.shape[0]
        self.targets = list(set(self.data[self.target_name]))
        self.n_targets = len(self.targets)
        self.target_dct = None
        self.target_dct_inv = None
        if shuffle_data:
            self.shuffle()
            self.shuffled = True
        else:
            self.shuffled = False
        self.scales = [np.ones(shape=(self. n_features, 1)), np.zeros(shape=(self.n_features, 1))]

    def _store_data_as_df(self, data, feature_names, target_name):
        """
        Store data array as a dataframe

        Parameters
        ----------
        data : array, shape = (n_samples, n_features + 1)
            Original dataset stored as a numpy array
        """
        X, y = data
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [''.join(['X', str(i)]) for i in range(n_features)]
        if target_name is None:
            if y.ndim == 1:
                target_name = ['y']
            else:
                target_name = [''.join(['y', str(i)]) for i in range(y.shape[1])]
        n_targets = len(target_name)
        colnames = feature_names + target_name
        data = pd.DataFrame(np.concatenate((X, y.reshape(n_samples, n_targets)), axis=1), columns=colnames)
        self.feature_names = feature_names
        self.target_name = target_name
        return data

    def int_enc(self):
        """
        Encode target values as integers for classification

        ...

        Returns
        -------
        y : array
            Integer encoded array of target values
        """
        new_targets = list(range(self.n_targets))
        self.target_dct = dict(zip(self.targets, new_targets))
        self.target_dct_inv = dict(zip(new_targets, self.targets))
        y = np.empty(shape=(self.n_samples, ))
        for i in range(self.n_samples):
            y[i] = self.target_dct[self.data[self.target_name][i]]
        return y

    def _extract_arrays(self, data=None):
        """
        Extract the feature and target values from dataframe

        ...

        Parameters
        ----------
        data : array (default=None)
            Dataset

        Returns
        -------
        X : array
            Array of feature values
        y : array
            Array of target values
        """
        if data is None:
            data = self.data
        X = np.array(data[self.feature_names])
        y = np.array(data[self.target_name]).flatten()
        return X, y

    def shuffle(self):
        """Shuffle dataset stored in self.data_df

        """
        self.data = self.data.sample(frac=1)
        self.shuffled = True

    def str(self):
        """
        Display the structure of the dataset


        """

        (n_obs, n_vars) = self.data.shape
        var_names = list(self.data)
        data_types = self.data.dtypes
        print('Data structure')
        print(" ".join([str(n_obs), 'obs. of', str(n_vars), 'variables:']))
        for i in range(n_vars):
            var_name = var_names[i]
            data_type = data_types[i]
            if isinstance(self.data[var_name][0], float):
                values = [str(np.round(val,3)) for val in self.data[var_name][:5]]
            else:
                values = [str(val) for val in self.data[var_name][:5]]
            print(" ".join([var_name, ":", str(data_type)] + values))

    def train_test_split(self, frac=0.8, inprop=True, random_state=None):
        """Split the dataset self.data_df into training and test sets


        Parameters
        ----------
        frac : float, optional (default=0.8)
            Fraction of the dataset assigned to the training set (with the remaining
            (1 - frac) assigned to the test set
        inprop : Boolean, optional (default=True)
            If inprop is true, samples are drawn in proportion to their abundance in
            the dataset

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

        # if inprop:
        #     train = self.data.groupby(self.target_name).apply(lambda x: x.sample(frac=frac))
        # else:
        #     train = self.data.sample(frac=frac)
        # test = self.data.loc[~self.data.index.isin(train.index), :]
        if random_state is not None:
            np.random.seed(random_state)
        train = self.data.sample(frac=frac)
        test = self.data.loc[~self.data.index.isin(train.index), :]
        X_train, y_train = self._extract_arrays(train)
        X_test, y_test = self._extract_arrays(test)

        return X_train, y_train, X_test, y_test

    def plot(self):
        """
        Plot the dataset

        """
        if self.n_features == 3:
            self.data.plot.scatter(x=self.feature_names[0], y=self.feature_names[1],
                                   z=self.feature_names[2], c=self.target_name[0],
                                   colormap='viridis', edgecolors='w', linewidths=0.5)
        else:
            self.data.plot.scatter(x=self.feature_names[0], y=self.feature_names[1],
                                   c=self.target_name[0], colormap='viridis',
                                   edgecolors='w', linewidths=0.5)
        # if self.n_features == 3:
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     scatter = ax.scatter(self.data[self.feature_names[0]],
        #                          self.data[self.feature_names[1]],
        #                          self.data[self.feature_names[2]],
        #                          c=self.data[self.target_name],
        #                          edgecolors='w', linewidths=0.5)
        #     ax.set_xlabel(self.feature_names[0])
        #     ax.set_ylabel(self.feature_names[1])
        #     ax.set_zlabel(self.feature_names[2])
            # legend = ax.legend(*scatter.legend_elements(),
            #                    loc="lower left",
            #                    title=self.target_name[0])
            # ax.add_artist(legend)
        # else:
        #     sns.scatterplot(self.data[self.feature_names[0]],
        #                     self.data[self.feature_names[1]],
        #                     hue=self.data[self.target_name],
        #                     edgecolors='w', linewidths=0.5)
            # fig, ax = plt.subplots()
            # scatter = ax.scatter(self.data[self.feature_names[0]],
            #                      self.data[self.feature_names[1]],
            #                      c=self.data[self.target_name],
                                 # c=self.int_enc(),
                                 # edgecolors='w', linewidths=0.5)
            # ax.set_xlabel(self.feature_names[0])
            # ax.set_ylabel(self.feature_names[1])
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
            self.data_df = self._store_data_as_df(data, self.feature_names, self.target_name)
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
            self.data_df = self._store_data_as_df(data, self.feature_names, self.target_name)
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
            X_subset = self.data[self.feature_names]
        else:
            X_subset = self.data[terms]
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
            self.data = self._store_data_as_df(data, feature_names, self.target_name)
        return X_new

    def add_features(self, X_new, feature_names=None):
        """
        Add new, user-defined features to the existing dataset

        ...

        Parameters
        ----------
        X_new : array (nrows = X.shape[0])
            New feature values for each existing sample
        feature_names : string (default = None)
            Names of new features

        Returns
        -------
        self
            Augmented dataset with new feature values included
        """
        n_new_features = X_new.shape[1]
        X, y = self._extract_arrays()
        if feature_names is None:
            feature_names = self.feature_names + [''.join(['NF', str(f)]) for f in range(n_new_features)]
        else:
            feature_names = self.feature_names + feature_names
        data = np.concatenate((X, X_new), axis=1), y
        self.data = self._store_data_as_df(data, feature_names, self.target_name)
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
        if self.data is None:
            raise NotImplementedError('Dataset not found (i.e. not yet generated)')
        n_samples = self.n_samples
        X, y = self._extract_arrays()
        white_noise = np.array(np.random.normal(size=n_samples)).reshape(n_samples, 1)
        X = np.hstack((X, white_noise))
        feature_names = self.feature_names + ['Noise']
        data = X, y
        self.data = self._store_data_as_df(data, feature_names, self.target_name)
        return

    def pca(self, X=None, plot=False):
        """
        Perform a principle components analysis of the stored dataset

        ...

        Parameters
        ----------
        X : array (default = None)
            Dataset on which PCA analysis is performed. If none is provided,
            PCA is performed on self.data
        plot : Boolean (default = False)
            Plot ranked principle components

        Returns
        -------
        Z : array
            Array of principle component vectors (stored as columns)
        lambda : list
            List of eigenvalues
        """
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

    def pairplot(self, data=None, vars=None, cont=False):
        """
        A matrix of scatter plots displaying the correlations between covariates in data

        ...

        Parameters
        ----------
        data : array (default = None)
            Input dataset. If data is None, self.data is taken as default
        vars : string (default = None)
            Subset of features in data to plot
        cont : Boolean (default = False)
            Plot contour plots above diagonal
        """
        sns.set(style='whitegrid')
        if data is None:
            data = self.data
        if vars is None:
            vars = self.feature_names
        if cont:
            g = sns.PairGrid(data, vars=vars, hue=self.target_name[0])
            g = g.map_lower(plt.scatter, edgecolors='w', linewidths=0.5)
            g = g.map_diag(sns.kdeplot, shade=True)
            g = g.map_upper(sns.kdeplot)
            g = g.add_legend()
        else:
            sns.pairplot(data=data, vars=vars, hue=self.target_name[0])
        plt.show()

    def boxplot(self, data=None):
        """
        Draw a boxplot of data grouped by target value

        ...

        """
        if data is None:
            data = self.data
        data.boxplot(by=self.target_name)
        plt.show()


if __name__ == "__main__":
    generator = DataGenerator()
    doughnut_data = generator.make_wheel(noise=None, n_targets=3)
    container = DataContainer(doughnut_data)
    container.plot()

