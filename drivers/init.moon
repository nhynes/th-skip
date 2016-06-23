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
  crit = with nn.ParallelCriterion!
    \add nn.LMCriterion!
    \add nn.LMCriterion!
    \cuda!

  state = _.defaults opts.savedState or {},
      gpuSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
      gpuNextSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
      gpuPrevSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
      crit: crit
      t: 0

  drivers = {}
  lazyDrivers = {}

  for i, driver in pairs {'train', 'val', 'snap'}
    drivers[i] = (...) -> lazyDrivers[i](...)
    lazyDrivers[i] = (...) ->
      lazyDrivers[i] = dofile(thisfile driver..'.moon')(model, workers, opts, state)
      lazyDrivers[i](...)

  table.unpack drivers

{ :init }
