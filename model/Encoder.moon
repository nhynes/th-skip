Encoder, parent = torch.class('Encoder', 'nn.Container')

Encoder.__init = (opts) =>
  -- opts = {:vocabSize, :w2v, :dim, :nRNNs}
  parent.__init(self)

  @lut = nn.LookupTableW2V(opts.w2v, opts.vocabSize, 2) -- UNK and </r> are learned

  rnnType = (opts.bidir == 1 and 'B' or '')..opts.rnnType\upper!
  @rnn = cudnn[rnnType](@lut.nOutput, opts.dim, opts.nEncRNNs)

  @embDim = @rnn.numDirections * opts.dim

  @means = nn.Identity!
  @stds = nn.Linear(@embDim, @embDim)
  @noise = with nn.Sequential!
    \add @stds
    \add nn.Exp!
    \add nn.HardTanh(0, 20)
    \add nn.Din!

  @encoder = with nn.Sequential!
    \add @lut
    \add @rnn
    \add nn.Select(1, -1)
    \add with nn.ConcatTable!
      \add @means
      \add @noise

  @sum = with nn.Sequential!
    \add nn.CAddTable!
    \add nn.Normalize(2)

  @modules = {@encoder, @sum}

Encoder.updateOutput = (input) =>
  [==[
  input: toks (N x sentlen, elems in [1, vocabSize])
  output: sentVecs (N x embDim)
  ]==]
  @encoder\forward(input)
  @output = @sum\forward(@encoder.output)
  return @output

Encoder.updateGradInput = (input, gradOutput) =>
  @sum\backward(@encoder.output, gradOutput)
  @sum.gradInput[1]\add(@means.output) -- enforce prior over mu
  @gradInput = @encoder\backward(input, @sum.gradInput)
  @gradInput
