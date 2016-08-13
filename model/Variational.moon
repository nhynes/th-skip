Variational, parent = torch.class('nn.Variational', 'nn.Container')

Variational.__init = (inDim, outDim)=>
  @inDim = inDim
  @outDim = outDim

  @var = with nn.ConcatTable!
    \add nn.Linear(inDim, outDim) -- mu
    \add with nn.Sequential!      -- sigma
      \add nn.Linear(inDim, outDim)
      \add nn.Exp!
      \add nn.Clamp(0, 10)

  @noise = torch.Tensor!
  @dMu = torch.Tensor!
  @dSigma = torch.Tensor!
  @output = torch.Tensor!

  @modules = {@var}

Variational.updateOutput = (input) =>
  {mu, sigma} = @var\forward(input)

  -- @fwds = @fwds or 0
  -- if @fwds % 1000 == 0
  --   print mu\mean!, sigma\mean!
  -- @fwds += 1

  with @noise\resizeAs(mu)
    \normal! if @train == true
    \zero! if @train ~= true
  @output\resizeAs(@noise)\copy(@noise)\cmul(sigma)\add(mu)
  @output

Variational.updateGradInput = (input, gradOutput) =>
  {mu, sigma} = @var.output

  gom = gradOutput\max!*0.001

  @dMu\resizeAs(mu)\copy(gradOutput)\add(gom/mu\max!, mu)

  with @dSigma\resizeAs(sigma)\copy(sigma)\add(1e-8)\cinv!\mul(-1)
    \add(1, sigma)
    \mul(gom/@dSigma\max!)
    \addcmul(@noise, gradOutput)

  @gradInput = @var\backward(input, {@dMu, @dSigma})
  @gradInput
