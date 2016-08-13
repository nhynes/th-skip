Din, parent = torch.class('nn.Din', 'nn.Module')
-- scales unit normal noise by standard deviations

-- Kingma, Diederik P., and Max Welling. “Auto-Encoding Variational Bayes.”
-- December 20, 2013. http://arxiv.org/abs/1312.6114.

Din.__init = =>
  parent.__init(self)
  @noise = torch.Tensor!
  @gvb = torch.Tensor!

Din.updateOutput = (input) =>
  -- input: pointwise stdevs, output: noise, scaled by stdevs
  @noise\randn(input\size!)
  return @output\resizeAs(input)\copy(input)\cmul(@noise)

Din.updateGradInput = (input, gradOutput) =>
  @gvb\resizeAs(input)\copy(input)\add(1e-8)\cinv!\mul(-1)\add(input)
  -- @gvb\mul(0.1*gradOutput\max!/@gvb\max!)
  return @gradInput\resizeAs(input)\copy(gradOutput)\cmul(@noise)\add(@gvb)
