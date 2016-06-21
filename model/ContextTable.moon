require 'torch'
require 'nn'

ContextTable, parent = torch.class('nn.ContextTable', 'nn.Module')
-- appends a context vector to every vector in a sequence

ContextTable.__init = (dimension) =>
  parent.__init(self)
  @gradInput = {torch.Tensor!, torch.Tensor!}

ContextTable.updateOutput = (input) =>
  [==[
    input: {base, context}
      base - seqlen x batchSize x m
      context - batchSize x n
    output: seqlen x batchSize x m+n
  ]==]
  base, ctx = input[1], input[2]

  seqlen, batchSize, baseDim = base\size(1), base\size(2), base\size(3)

  ctxDim = ctx\size(2)
  repCtx = ctx\view(1, batchSize, ctxDim)\expand(seqlen, batchSize, ctxDim)

  with @output
    \resize(seqlen, batchSize, baseDim+ctxDim)
    \narrow(3, 1, baseDim)\copy(base)
    \narrow(3, baseDim+1, ctxDim)\copy(repCtx)

ContextTable.updateGradInput = (input, gradOutput) =>
  base, ctx = input[1], input[2]
  baseDim, ctxDim = base\size(3), ctx\size(2)

  @gradInput[1]\resizeAs(base)\copy(gradOutput\narrow(3, 1, baseDim))

  gradCtx = gradOutput\narrow(3, baseDim+1, ctxDim)
  @gradInput[2]\resizeAs(ctx)\view(1, -1, ctxDim)\sum(gradCtx, 1)

  @gradInput
