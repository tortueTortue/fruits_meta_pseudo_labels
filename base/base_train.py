import tensorflow as tf


class BaseTrain:
    def __init__(self, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data = data


    def train(self):
        for cur_epoch in range(self.config.num_epochs):
            self.train_epoch()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
