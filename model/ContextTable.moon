ContextTable, parent = torch.class('nn.ContextTable', 'nn.Module')
-- appends a context vector to every vector in a sequence

ContextTable.__init = (lpad) =>
  parent.__init(self)
  @gradInput = {torch.Tensor!, torch.Tensor!, torch.Tensor!}

ContextTable.updateOutput = (input) =>
  [==[
    input: {base, context, lpad}
      base - (seqlen x batchSize x m)
      context - (batchSize x n)
      lpad - left-padding for base (p x batchSize x m)
    output: seqlen x batchSize x m+n
  ]==]
  {base, ctx, lpad} = input

  seqlen, batchSize, baseDim = base\size(1), base\size(2), base\size(3)

  ctxDim = ctx\size(2)

  padLen = lpad ~= nil and lpad\size(1) or 0
  padSeqlen = padLen + seqlen

  @output = @output\resize(padSeqlen, batchSize, baseDim+ctxDim)

  with @output\narrow(3, 1, baseDim)
    \narrow(1, 1, padLen)\copy(lpad) if lpad ~= nil
    \narrow(1, padLen+1, seqlen)\copy(base)

  repCtx = ctx\view(1, batchSize, ctxDim)\expand(padSeqlen, batchSize, ctxDim)
  @output\narrow(3, baseDim+1, ctxDim)\copy(repCtx)

  @output

ContextTable.updateGradInput = (input, gradOutput) =>
  {base, ctx, lpad} = input

  seqlen, baseDim, ctxDim = base\size(1), base\size(3), ctx\size(2)
  padLen = lpad ~= nil and lpad\size(1) or 0

  gradBase = gradOutput\narrow(1, padLen+1, seqlen)\narrow(3, 1, baseDim)
  @gradInput[1]\resizeAs(base)\copy(gradBase)

  gradCtx = gradOutput\narrow(3, baseDim+1, ctxDim)
  @gradInput[2] = @gradInput[2]\sum(gradCtx, 1)\viewAs(ctx)

  if lpad ~= nil
    gradPad = gradOutput\narrow(1, 1, padLen)\narrow(3, 1, baseDim)
    @gradInput[3]\resizeAs(lpad)\copy(gradPad)

  @gradInput
