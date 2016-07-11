SeqMaskedCrossEntropyCriterion, parent = torch.class('nn.SeqMaskedCrossEntropyCriterion', 'nn.Criterion')

SeqMaskedCrossEntropyCriterion.__init = (weights) =>
  parent.__init(self)
  @crit = nn.MaskedCrossEntropyCriterion(weights)

SeqMaskedCrossEntropyCriterion.updateOutput = (input, target) =>
  [==[
  input: per-token class probabilities (seqLen x batchSize x nClasses)
  target: per-token class labels (batchSize x seqLen, elems in [0, nTags]; 0 is pad)
  output: loss
  Note: Forward and backward must be called alternately.
  ]==]
  @gradInput\resizeAs(input)
  @output = 0
  assert input\size(1) == target\size(2)
  for i=1,input\size(1)
    stepInp = input[i]
    stepTgt = target\select(2, i)

    @output += @crit\forward(stepInp, stepTgt)
    @gradInput[i] = @crit\backward(stepInp, stepTgt)

  @output

SeqMaskedCrossEntropyCriterion.updateGradInput = (input, target) => @gradInput
