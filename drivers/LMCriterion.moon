LMCriterion, parent = torch.class('nn.LMCriterion', 'nn.Criterion')

LMCriterion.__init = =>
  parent.__init(self)
  @sm = cudnn.LogSoftMax!
  @crit = nn.ClassNLLCriterion!
  @mask = torch.Tensor!
  @mTarget = torch.Tensor!
  @mInput = torch.Tensor!

LMCriterion.updateOutput = (input, target) =>
  @output = 0
  @gradInput\resizeAs(input)
  assert input\size(1) == target\size(2)
  for i=1,input\size(1)
    stepInp = input[i]
    stepTgt = target\select(2, i)

    stepPreds = @sm\forward(stepInp)

    -- assign probability 1 (log 1 = 0) to everything so that there's no loss
    @mask\ne(stepTgt, 0)
    predsMask = @mask\view(-1, 1)\expandAs(stepPreds) -- zero target = zero row
    @mInput\resizeAs(stepPreds)\copy(stepPreds)\cmul(predsMask)

    -- make the target index nonzero so that it isn't off the end of the tensor
    @mask\eq(stepTgt, 0)
    @mTarget\resizeAs(stepTgt)\copy(stepTgt)\maskedFill(@mask, 1)

    @output += @crit\forward(@mInput, @mTarget)
    @crit\backward(@mInput, @mTarget)
    @sm\backward(stepInp, @crit.gradInput)
    @gradInput[i] = @sm.gradInput

  @output

LMCriterion.updateGradInput = (input, target) => @gradInput
