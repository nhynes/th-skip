require 'hdf5'
_ = require 'moses'
threads = require 'threads'
threads.serialization 'threads.sharedserialize'

export dataLoader

UNK, SOR, EOR, EOS = 1, 2, 3, 4

-- import dofile from require 'moonscript'
-- import thisfile from require 'paths'

-- dofile(thisfile 'DataLoader.moon')

loadPartition = (dsH5, partition, opts) ->
  toks = dsH5\read('/toks_'..partition)\all!
  toks\maskedFill(toks\gt(opts.vocabSize), UNK)
  {
    -- ids: dsH5\read('/ids_'..partition)\all!
    -- rlens: dsH5\read('/rlens_'..partition)\all!
    toks: toks
    slens: dsH5\read('/slens_'..partition)\all!
    rbps: dsH5\read('/rbps_'..partition)\all!
  }

init = (opts) ->
  {:nworkers, :seed} = opts

  dsH5 = hdf5.open(opts.dataset, 'r')
  dataTrain = loadPartition(dsH5, 'train', opts)
  dataVal = loadPartition(dsH5, 'val', opts)
  dsH5\close!

  loaderOpts = _.omit(opts, 'savedState')

  if nworkers > 0
    return threads.Threads nworkers,
      ->
        require 'moonscript'
        require 'torch'
        require 'cutorch'
        require 'loader.DataLoader',
      (tid) ->
        math.randomseed seed+tid
        torch.manualSeed seed+tid
        dataLoader = DataLoader(dataTrain, dataVal, loaderOpts)
  else -- single threaded data loading. useful for debugging
    require 'loader.DataLoader'
    dataLoader = DataLoader(dataTrain, dataVal, loaderOpts)
    return {
      addjob: (f1, f2) => f2(f1())
      synchronize: =>
      terminate: =>
    }

{ :init }
