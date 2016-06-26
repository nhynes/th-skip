require 'torch'
require 'cutorch'
require 'nn'
require 'rnn'
require 'cunn'
_ = require 'moses'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'LMCriterion.moon')

init = (model, workers, opts) ->
  crit = nil
  if opts.decoding ~= ''
    crit = nn.LMCriterion!
  else
    crit = with nn.ParallelCriterion!
      \add nn.LMCriterion!
      \add nn.LMCriterion!

  state = _.defaults opts.savedState or {},
      t: 0
      crit: crit\cuda!

  gpuSents = torch.CudaTensor(opts.batchSize, opts.sentlen)
  state.gpuSents = gpuSents
  if opts.decoding ~= ''
    state.prepBatch = (batchSents) ->
      gpuSents\resize(batchSents\size!)\copy(batchSents)
      trimEOS = gpuSents[{{}, {1, -2}}]
      {gpuSents[{{}, {2, -1}}], trimEOS}, trimEOS
  else
    gpuNextSents = torch.CudaTensor(opts.batchSize, opts.sentlen)
    gpuPrevSents = torch.CudaTensor(opts.batchSize, opts.sentlen)

    state.prepBatch = (batchSents, batchPrevSents, batchNextSents) ->
      gpuSents\resize(batchSents\size!)\copy(batchSents)
      gpuPrevSents\resize(batchPrevSents\size!)\copy(batchPrevSents)
      gpuNextSents\resize(batchNextSents\size!)\copy(batchNextSents)

      input = {gpuSents, gpuPrevSents[{{}, {1, -2}}], gpuNextSents[{{}, {1, -2}}]}
      target = {gpuPrevSents[{{}, {2, -1}}], gpuNextSents[{{}, {2, -1}}]}

      state.gpuPrevSents = gpuPrevSents
      state.gpuNextSents = gpuNextSents

      input, target


  drivers = {}
  lazyDrivers = {}

  for i, driver in pairs {'train', 'val', 'snap'}
    drivers[i] = (...) -> lazyDrivers[i](...)
    lazyDrivers[i] = (...) ->
      lazyDrivers[i] = dofile(thisfile driver..'.moon')(model, workers, opts, state)
      lazyDrivers[i](...)

  table.unpack drivers

{ :init }
