Decoder, parent = torch.class('Decoder', 'nn.Container')

Decoder.__init = (opts, sharedModules) =>
  -- opts = {:wembDim, :embDim, :dim, :nRNNs, :vocabSize, :w2v}
  parent.__init(self)

  if sharedModules.lut ~= nil
    @lut = sharedModules.lut\clone('weight', 'gradWeight')
  else
    @lut = nn.LookupTableW2V(opts.w2v, opts.vocabSize, 2) -- UNK, </r>, </s>

  if sharedModules.dec ~= nil
    @wordDec = sharedModules.dec\clone('weight', 'gradWeight', 'bias', 'gradBias')
  else
    @wordDec = nn.TemporalConvolution(opts.dim, @lut.nIndex, 1)
    -- using cudnn causes a ~5% drop in accuracy but is ~.15s/batch faster

  @rnn = cudnn.GRU(@lut.nOutput + opts.embDim, opts.dim, opts.nRNNs)

  @model = with nn.Sequential!
    \add with nn.ParallelTable!
      \add @lut
      \add nn.Identity!
    \add nn.ContextTable!
    \add @rnn
    \add @wordDec

  @modules = {@model}

  collectgarbage!

Decoder.updateOutput = (input) =>
  [==[
  input: {toks, sentVecs}
    toks (N x sentlen; elems in [1, vocabSize], 0 is pad)
    sentVecs (N x embDim)
  output: preds (sentlen x N x vocabSize)
  ]==]
  @output = @model\forward(input)
  @output

Decoder.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
