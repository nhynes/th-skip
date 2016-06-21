require 'cudnn'

BGRU, parent = torch.class('cudnn.BGRU', 'cudnn.RNN')

BGRU.__init = (inputSize, hiddenSize, numLayers, batchFirst) =>
    parent.__init(self, inputSize, hiddenSize, numLayers, batchFirst)
    @bidirectional = 'CUDNN_BIDIRECTIONAL'
    @mode = 'CUDNN_GRU'
    @numDirections = 2
    self\reset!
