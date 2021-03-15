"""
Meta Pseudo Labels Network
"""
import tensorflow as tf

class MetaPseudoLabelsNet(tf.Module):


    def __init__(self, name="MetaPseudoLabels", student, teacher):
        super(MetaPseudoLabelsNet, self).__init__(name=name)
        self.student = student
        self.teacher = teacher
        
    def __call__(self, x):
        y = self.student(x)
        y_ = self.teacher(x)
        return 0