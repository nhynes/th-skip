require 'torch'
require 'cutorch'
require 'optim'

(model, workers, opts, state) ->
  {:gpuSents, :gpuPrevSents, :gpuNextSents, :crit} = state

  state.trainLoss = 0

  optimState = learningRate: opts.lr

  if opts.optim == 'sgd'
    with optimState
      .beta = 0.9
      .beta2 = 0.999
      .epsilon = 1e-8
  else
    with optimState
      .momentum = 0.9
      .nesterov = true
      .dampening = 0
      .weightDecay = 5e-4

  optimizer = opts.optim == 'sgd' and optim.sgd or optim.adam
  params, gradParams = model\getParameters!

  f = -> crit.output, gradParams

  doTrain = (batchSents, batchPrevSents, batchNextSents) ->
    state.t += 1

    gpuSents\resize(batchSents\size!)\copy(batchSents)
    gpuPrevSents\resize(batchPrevSents\size!)\copy(batchPrevSents)
    gpuNextSents\resize(batchNextSents\size!)\copy(batchNextSents)

    input = {gpuSents, gpuPrevSents, gpuNextSents}
    target = {gpuPrevSents\t!\sub(2, -1), gpuNextSents\t!\sub(2, -1)}

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
