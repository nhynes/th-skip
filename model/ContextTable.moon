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
      base - batchSize x seqLen x m
      context - batchSize x n
    output: batchSize x seqlen x m+n
  ]==]
  base, ctx = input[1], input[2]

  batchSize, seqlen, baseDim = base\size(1), base\size(2), base\size(3)

  ctxDim = ctx\size(2)
  repCtx = ctx\view(batchSize, 1, ctxDim)\expand(batchSize, seqlen, ctxDim)

  with @output
    \resize(batchSize, seqlen, baseDim+ctxDim)
    \narrow(3, 1, baseDim)\copy(base)
    \narrow(3, baseDim+1, ctxDim)\copy(repCtx)

ContextTable.updateGradInput = (input, gradOutput) =>
  base, ctx = input[1], input[2]
  baseDim, ctxDim = base\size(3), ctx\size(2)

  @gradInput[1]\resizeAs(base)\copy(gradOutput\narrow(3, 1, baseDim))

  gradCtx = gradOutput\narrow(3, baseDim+1, ctxDim)
  @gradInput[2]\resizeAs(ctx)\view(-1, 1, ctxDim)\sum(gradCtx, 2)

  @gradInput
