Encoder, parent = torch.class('Encoder', 'nn.Container')

Encoder.__init = (opts) =>
  -- opts = {:vocabSize, :w2v, :dim, :nRNNs}
  parent.__init(self)

  @lut = nn.LookupTableW2V(opts.w2v, opts.vocabSize, 2) -- UNK and </r> are learned

  rnnType = (opts.bidir == 1 and 'B' or '')..opts.rnnType\upper!
  @rnn = cudnn[rnnType](@lut.nOutput, opts.dim, opts.nEncRNNs)

  @embDim = @rnn.numDirections * opts.dim

  @model = with nn.Sequential!
    \add @lut
    \add @rnn
    \add nn.Select(1, -1)
    \add nn.Normalize(2)

  @modules = {@model}

  collectgarbage!

Encoder.updateOutput = (input) =>
  [==[
  input: toks (N x sentlen, elems in [1, vocabSize])
  output: sentVecs (N x embDim)
  ]==]
  @output = @model\forward(input)
  @output

Encoder.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
