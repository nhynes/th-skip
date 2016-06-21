require 'dpnn'

(model, workers, opts, state) ->
  OUTFILE_TMP = opts.snapfile..'_i%s_v%.3f.t7'

  serializer = nn.Serial(model)\mediumSerial!
  bestPerf = -math.huge

  ->
    if state.valPerf > bestPerf
      bestPerf = state.valPerf
      outfile = string.format OUTFILE_TMP, state.t, bestPerf
      print 'Saving model to '..outfile..'...'
      torch.save outfile, serializer
