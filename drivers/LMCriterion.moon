require 'torch'
require 'nn'

LMCriterion, parent = torch.class('nn.LMCriterion', 'nn.Criterion')

LMCriterion.__init = =>
  parent.__init(self)
  @crit = nn.ClassNLLCriterion!
  @mask = torch.Tensor!
  @maskedTarget = torch.Tensor!
  @zeroedInput = torch.Tensor!

LMCriterion.updateOutput = (input, target) =>
  -- assign probability 1 to everything so that there's no loss
  @mask\ne(target, 0)
  @zeroedInput\resizeAs(input)\copy(input)\cmul(@mask\view(-1, 1)\expandAs(input))

  -- just make the target nonzero so that the index isn't off the end of the tensor
  @mask\eq(target, 0)
  @maskedTarget\resizeAs(target)\copy(target)\maskedFill(@mask, 1)

  @output = @crit\forward(@zeroedInput, @maskedTarget)
  @output

LMCriterion.updateGradInput = (input, target) =>
  @gradInput = @crit\backward(@zeroedInput, @maskedTarget)
  @gradInput
