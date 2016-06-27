require 'torch'
require 'cutorch'
require 'optim'

(model, workers, opts, state) ->
  {:prepBatch, :crit, :optimState} = state

  state.trainLoss = 0

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

    model\forward(input)
    state.trainLoss += crit\forward(model.output, target)

    model\zeroGradParameters!
    crit\backward(model.output, target)
    model\backward(input, crit.gradInput)
    optimizer(f, params, optimState)

    if state.t % opts.dispfreq == 0
      print string.format('Train loss: %g', state.trainLoss/opts.dispfreq)
      state.trainLoss = 0

    assert params\storage! == model\parameters![1]\storage!

  ->
    model\training!
    workers\addjob((-> dataLoader\makebatch!), doTrain)
