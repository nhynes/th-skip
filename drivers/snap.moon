require 'dpnn'
_ = require 'moses'

(model, workers, opts, state) ->
  OUTFILE_TMP = opts.snapfile..'_i%s_v%.3f.t7'

  serializer = nn.Serial(model)\mediumSerial!
  state.bestLoss = state.bestLoss or math.huge

  ->
    startedSaving = opts.saveafter == -1 or state.t > opts.saveafter
    if state.valLoss <= state.bestLoss and state.t >= opts.saveafter
      state.bestLoss = state.valLoss

      outfile = string.format OUTFILE_TMP, state.t, state.bestLoss

      saveState = _.pick(state, 't', 'optimState', 'bestLoss')

      print 'Saving model to '..outfile..'...'
      model\training!
      torch.save outfile, {opts: opts, model: serializer, state: saveState}
