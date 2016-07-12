SIPCriterion, parent = torch.class('nn.SIPCriterion', 'nn.Criterion')

-- sequential in-place criterion: forwards and backwards sequentially in one go
-- Note: a particular gradInput is only valid for the previous call to forward

SIPCriterion.__init = (crit) =>
  parent.__init(self)
  @crit = crit

SIPCriterion.updateOutput = (input, target) =>
  @gradInput\resizeAs(input)
  @output = 0
  assert input\size(1) == target\size(2)
  for i=1,input\size(1)
    stepInp = input[i]
    stepTgt = target\select(2, i)

    @output += @crit\forward(stepInp, stepTgt)
    @gradInput[i] = @crit\backward(stepInp, stepTgt)

  @output

SIPCriterion.updateGradInput = (input, target) => @gradInput
