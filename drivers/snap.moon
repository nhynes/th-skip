require 'dpnn'

(model, workers, opts, state) ->
  OUTFILE_TMP = opts.snapfile..'_i%s_v%.3f.t7'

  serializer = nn.Serial(model)\mediumSerial!
  bestLoss = math.huge

  ->
    if state.valLoss <= bestLoss
      bestLoss = state.valLoss
      outfile = string.format OUTFILE_TMP, state.t, bestLoss
      print 'Saving model to '..outfile..'...'
      torch.save outfile, serializer
