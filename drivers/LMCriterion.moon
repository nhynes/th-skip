require 'torch'
require 'nn'

LMCriterion, parent = torch.class('nn.LMCriterion', 'nn.Criterion')

LMCriterion.__init = =>
  parent.__init(self)
  @crit = nn.ClassNLLCriterion!
  @mask = torch.Tensor!
  @mTarget = torch.Tensor!
  @mInput = torch.Tensor!

LMCriterion.updateOutput = (input, target) =>
  @output = 0
  @gradInput\resizeAs(input)
  assert input\size(1) == target\size(1)
  for i=1,input\size(1)
    stepInp = input\select(1, i)
    stepTgt = target\select(1, i)

    -- assign probability 1 to everything so that there's no loss
    @mask\ne(stepTgt, 0)
    @mInput\resizeAs(stepInp)\copy(stepInp)\cmul(@mask\view(-1, 1)\expandAs(stepInp))

    -- just make the target nonzero so that the index isn't off the end of the tensor
    @mask\eq(stepTgt, 0)
    @mTarget\resizeAs(stepTgt)\copy(stepTgt)\maskedFill(@mask, 1)

    @output += @crit\forward(@mInput, @mTarget)
    @crit\backward(@mInput, @mTarget)
    @gradInput\select(1, i)\copy(@crit.gradInput)

  @output

LMCriterion.updateGradInput = (input, target) =>
  -- @gradInput\resizeAs(input)
  -- for i=1,input\size(1)
  --   stepInp = input\select(1, i)
  --   stepTgt = target\select(1, i)
  --
  --   -- assign probability 1 to everything so that there's no loss
  --   @mask\ne(stepTgt, 0)
  --   @mInput\resizeAs(stepInp)\copy(stepInp)\cmul(@mask\view(-1, 1)\expandAs(stepInp))
  --
  --   -- just make the target nonzero so that the index isn't off the end of the tensor
  --   @mask\eq(stepTgt, 0)
  --   @mTarget\resizeAs(stepTgt)\copy(stepTgt)\maskedFill(@mask, 1)
  --
  --   @crit\backward(@mInput, @mTarget)
  --   @gradInput\select(1, i)\copy(@crit.gradInput)
  @gradInput
