import numpy as np

class LogisticRegressionModel:
    """
    Logistic Regression ML model
    Matrix-based implementation from scratch using NumPy.

    FEATURES:
        - Adjustable learning rate and number of epochs
        - Gradient descent optimization
        - Supports any number of features (multivariate)
        - Vectorized for fast, batched probability predictions and classifications
        - Automatically scales input features (standardization)
        - Tracks training and optional test performance (Binary Cross Entropy and Mean Accuracy)

    PUBLIC METHODS:
        __init__             - Initializes model with hyperparameters and metric tracking option
        fit                  - Trains model using gradient descent, with optional test set tracking
        predict_probs        - Returns predicted probabilities for each row of input feature matrix
        predict              - Returns classifications for each row of input feature matrix
        get_params           - Returns learned weights as unscaled slopes and intercept
        get_training_history - Returns tracked metrics over training epochs
    """

    def __init__(self, alpha=0.01, epochs=1_000, track_metrics=True):
        # Ensure reasonable learning rate range
        if not (1e-6 <= alpha <= 1.0):
            raise ValueError('Alpha must be between 1e-6 and 1.0')

        # Ensure reasonable training epochs range
        if not (1 <= epochs <= 100_000):
            raise ValueError('Epochs must be between 1 and 100,000')

        # Set learning rate and epochs
        self.alpha = alpha
        self.epochs = epochs

        # Set tracking variables
        self.track_metrics = track_metrics

        # Initialize other model variables
        self._reset()


    def _reset(self):
        self.x_b = None
        self.y = None
        self.theta = None
        self.n = None
        self.mean = None
        self.std = None

        self.losses = None
        self.test_accuracies = None
        self.train_accuracies = None


    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def _prepare_feature_matrix(x_prep, _mean=None, _std=None):
        # Ensure x is a numpy array
        x = np.array(x_prep, copy=True)

        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Scale feature matrix
        mean = None
        std = None

        if _mean is None or _std is None:
            mean = x.mean(axis=0)
            std = x.std(axis=0)
        else:
            mean = _mean
            std = _std

        x = (x - mean) / std

        # Create bias column for feature matrix
        bias_col = np.ones((x.shape[0], 1))
        x = np.hstack((bias_col, x))

        return x, mean, std


    @staticmethod
    def _prepare_target_vector(y_prep):
        # Ensure y is a numpy array
        y = np.array(y_prep, copy=True)

        # Ensure y is a column vector
        if y.ndim != 2 or y.shape[1] != 1:
            y = np.reshape(y, (-1, 1))

        # Ensure y only contains values 0 and 1 (True/False Classification)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Target vector y must contain only 0s and 1s.")

        return y


    def _prepare_train_test_set(self, x, y, _mean=None, _std=None):
        # Prepare matrices
        x, mean, std = self._prepare_feature_matrix(x, _mean=_mean, _std=_std)
        y = self._prepare_target_vector(y)

        # Ensure x and y agree in sample size
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have the same number of samples/rows')

        n = x.shape[0]

        return x, y, mean, std, n


    def _is_trained(self):
        return self.theta is not None


    @staticmethod
    def _binary_cross_entropy(y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


    def fit(self, x_train, y_train, x_test=None, y_test=None):
        # Reset class for new training
        self._reset()

        # Prepare training set
        self.x_b, self.y, self.mean, self.std, self.n = self._prepare_train_test_set(x_train, y_train)

        # Initialize theta
        self.theta = np.zeros((self.x_b.shape[1], 1))

        # Run training and gradient decent for epochs
        print(f"Starting training for {self.epochs} epochs...")
        losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(self.epochs):
            # Calculate predictions
            y_hat = LogisticRegressionModel._sigmoid(self.x_b @ self.theta)
            # Get error vector
            err = y_hat - self.y
            # Calculate gradient
            gradient = (1 / self.n) * self.x_b.T @ err
            # Update weights
            self.theta -= self.alpha * gradient
            # Track metrics
            if self.track_metrics and (epoch % 100 == 0 or epoch == self.epochs - 1):
                # Binary Cross-Entropy loss
                loss = float(LogisticRegressionModel._binary_cross_entropy(self.y, y_hat))
                losses.append(loss)

                # Train accuracy
                train_accuracy = self._score(self.y, y_hat)
                train_accuracies.append(train_accuracy)

                # Test accuracy
                test_accuracy = None
                if x_test is not None and y_test is not None:
                    x_test_prep, y_test_prep, *_ = self._prepare_train_test_set(x_test, y_test, self.mean, self.std)
                    y_hat_test = LogisticRegressionModel._sigmoid(x_test_prep @ self.theta)
                    test_accuracy = self._score(y_test_prep, y_hat_test)
                    test_accuracies.append(test_accuracy)

                print(f"Epoch {epoch} Loss: {loss} Train Accuracy: {train_accuracy} Test Accuracy: {test_accuracy}")

        # Training complete
        print(f"Training complete!")
        print(f"Loss: {losses[-1]} Train Accuracy: {train_accuracies[-1]} Test Accuracy: {test_accuracies[-1]}")
        self.losses = losses
        self.train_accuracies = train_accuracies
        self.test_accuracies = test_accuracies


    def predict_prob(self, x):
        # Prepare feature matrix
        x, *_ = self._prepare_feature_matrix(x, self.mean, self.std)

        # Get predicted probabilities
        y_hat = LogisticRegressionModel._sigmoid(x @ self.theta)
        return y_hat.flatten()


    def predict(self, x):
        # Get predicted probabilities
        probs = self.predict_prob(x)

        # Round probabilities to binary classification (0 or 1)
        predictions = (probs >= 0.5).astype(int)
        return predictions.flatten()


    @staticmethod
    def _score(y, y_hat):
        # Round probabilities to binary classification (0 or 1)
        predictions = (y_hat >= 0.5).astype(int)

        # Get mean accuracy
        accuracy = np.mean(predictions == y)
        return accuracy


    def get_params(self):
        # Ensure that model was trained
        if not self._is_trained():
            return None

        # Flatten theta
        theta_scaled = self.theta.flatten()

        # Separate bias and weights
        theta_bias = theta_scaled[0]
        theta_weights = theta_scaled[1:]

        # Unscale weights (slopes)
        unscaled_weights = theta_weights / self.std

        # Unscale y-intercept
        unscaled_bias = theta_bias - np.sum(unscaled_weights * self.mean)

        # Form unscaled theta
        theta_unscaled = np.hstack([unscaled_bias, unscaled_weights])

        return {
            "theta": theta_unscaled.flatten(),
            "bias": unscaled_bias,
            "weights": unscaled_weights.copy()
        }


    def get_training_history(self):
        # Ensure that model was trained
        if not self._is_trained():
            print("Call fit method to train model and record metrics.")
            return None

        # Ensure that metrics were tracked
        if not self.track_metrics:
            print("No metrics tracked. Set track_metrics=True to record metrics.")
            return None

        return {
            "loss": self.losses.copy(),
            "train_accuracy": self.train_accuracies.copy(),
            "test_accuracy": self.test_accuracies.copy(),
        }
