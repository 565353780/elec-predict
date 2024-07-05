from elec_predict.Model.Layer.wave_queue import WaveQueue

class WaveStates:
    def __init__(self, num_blocks, num_layers):
        self.queues = [WaveQueue(2 ** j) for _ in range(num_blocks) for j in range(num_layers)]

    def init(self, layer, x):
        self.queues[layer].init(x)

    def enqueue(self, layer, x):
        self.queues[layer].enqueue(x)

    def dequeue(self, layer):
        return self.queues[layer].dequeue()

    def clear_buffer(self):
        for q in self.queues:
            q.clear_buffer()


