require 'torch'
require 'dpnn'

DontTrain, parent = torch.class('nn.DontTrain', 'nn.Decorator')

DontTrain.__init = (module) =>
  parent.__init(self, module)

DontTrain.accGradParameters = (input, gradOutput, scale) => -- noop

DontTrain.parameters = =>
