class FastPredict:

    def __init__(self, estimator, input_fn, checkpoint_path, params):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn
        self.checkpoint_path = checkpoint_path
        self.params = params
        self.next_features = None
        self.batch_size = None
        self.predictions = None
        self.channel = params['num_channels']

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features.
            IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list
            (i.e. predict([my_feature]), not predict(my_feature)
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            print(self.batch_size)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(
                    self._create_generator,
                    (None, self.channel) + self.params['input_shape']),
                checkpoint_path=self.checkpoint_path)
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size."
                             " First-batch:" + str(self.batch_size) +
                             " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")
