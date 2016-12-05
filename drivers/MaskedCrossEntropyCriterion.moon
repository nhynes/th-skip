MaskedCrossEntropyCriterion, parent = torch.class('nn.MaskedCrossEntropyCriterion', 'nn.Criterion')

PLACEHOLDER_CLASS = 1

MaskedCrossEntropyCriterion.__init = (weights, sizeAverage=true) =>
  parent.__init(self)
  @sm = cudnn.LogSoftMax!
  @crit = nn.ClassNLLCriterion(weights, false) -- no size avg
  @mask = torch.ByteTensor!
  @maskByte = torch.ByteTensor!
  @mTarget = torch.Tensor!
  @sizeAverage = sizeAverage
  @nTargets = 0

MaskedCrossEntropyCriterion.type = (t) =>
  if t\find('Cuda') ~= nil
    nn.Module.type(self, t)
    @maskByte = @maskByte\cudaByte!

MaskedCrossEntropyCriterion.updateOutput = (input, target) =>
  [==[
  input: un-normalized probabilities (N x nClasses)
  target: targets (N, elems in [0, nClasses]; 0 is padding)
  output: loss
  Note: Forward and backward must be called alternately.
  ]==]
  probs = @sm\forward(input)

  @mask\eq(target, 0)
  @maskByte\resize(@mask\size!)\copy(@mask)
  @nTargets = @mask\numel! - @mask\sum!

  -- use the first class as a placeholder
  -- don't bother saving orig masked probs because gradInput will be zero
  @mTarget\resizeAs(target)\copy(target)\maskedFill(@maskByte, PLACEHOLDER_CLASS)

  -- assign P=1 (log 1 = 0) to the first class in masked positions to remove loss
  probs\select(2, PLACEHOLDER_CLASS)\maskedFill(@maskByte, 0)

  @output = @crit\forward(probs, @mTarget)
  @output /= @nTargets if @sizeAverage and @nTargets > 0

  @output

MaskedCrossEntropyCriterion.updateGradInput = (input, target) =>
  @gradInput\resizeAs(input)

  @crit\backward(@sm.output, @mTarget)

  -- no loss, no gradient!
  @crit.gradInput\select(2, PLACEHOLDER_CLASS)\maskedFill(@maskByte, 0)

  @gradInput = @sm\backward(input, @crit.gradInput)
  @gradInput\div(@nTargets) if @sizeAverage and @nTargets > 0

  @gradInput
