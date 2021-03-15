from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.metrics import loss, accuracy

class MLPTrainer(BaseTrain):
    def __init__(self, model, data, config,logger):
        super(MLPTrainer, self).__init__(model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        acc = np.mean(accs)

        
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # SAVE MODEL

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def train_model(self):

        for epoch in range(self.config.num_epochs):
            """ T R A N I N G """
            for i, l in self.data.train_set:
                train(self.model, i, l, learning_rate=0.1)

                Ws.append(model.w.numpy())
                bs.append(model.b.numpy())
                current_loss = loss(l, model(i))
            
            self.data.train_set.shuffle(buffer_size=100)

            """ V A L I D A T I O N"""
            print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
                (epoch, Ws[-1], bs[-1], current_loss))
            print(f"Accuracy : {accuracy(model, self.data.val_set)}")

    def train(self, model, images, labels, learning_rate):

        with tf.GradientTape() as t:
            current_loss = loss(labels, model(images))

        dw, db = t.gradient(current_loss, [model.w, model.b])

        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)