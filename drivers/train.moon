require 'optim'
require 'sys'

(model, workers, opts, state) ->
  {:prepBatch, :crit, :optimState} = state

  state.trainLoss = 0
  state.timer = 0

  dispfreq = opts.dispfreq

  if optimState == nil
    optimState =
      learningRate: opts.lr

    if opts.optim == 'sgd'
      with optimState
        .momentum = 0.9
        .nesterov = true
        .dampening = 0
    else
      with optimState
        .beta = 0.9
        .beta2 = 0.999
        .epsilon = 1e-8

    state.optimState = optimState

  optimizer = opts.optim == 'sgd' and optim.sgd or optim.adam
  params, gradParams = model\getParameters!

  f = -> crit.output, gradParams

  doTrain = (...) ->
    state.t += 1

    input, target = prepBatch(...)

    sys.tic!
    model\forward(input)
    state.trainLoss += crit\forward(model.output, target)

    model\zeroGradParameters!
    crit\backward(model.output, target)
    model\backward(input, crit.gradInput)
    optimizer(f, params, optimState)
    state.timer += sys.toc!

    if state.t % opts.dispfreq == 0
      print string.format('Train loss: %g\t(%.2f)', state.trainLoss/dispfreq, state.timer/dispfreq)
      state.trainLoss = 0
      state.timer = 0

    paramsStore = params\storage!
    assert paramsStore == model\parameters![1]\storage!
    if opts.decoding ~= ''
      assert paramsStore == model.decoder\parameters![1]\storage!
      assert paramsStore ~= model.decoder.lut.weight\storage!
      assert paramsStore ~= model.encoder.lut.weight\storage!
      assert paramsStore ~= model.encoder.rnn.weight\storage!

  ->
    model\training!
    workers\addjob((-> dataLoader\makebatch!), doTrain)
