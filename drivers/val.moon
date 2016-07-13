(model, workers, opts, state) ->
  {:prepBatch, :crit} = state

  valBatches = math.ceil(50000 / opts.batchSize)

  ->
    model\evaluate!

    randState = torch.getRNGState!
    torch.manualSeed(1234)

    valLoss = 0

    for i=1,valBatches
      workers\addjob (-> dataLoader\makebatch 'val', torch.random!),
        (...) ->
          input, target = prepBatch(...)
          model\forward(input)
          valLoss += crit\forward(model.output, target)

    workers\synchronize!

    torch.setRNGState(randState)

    valLoss /= valBatches

    state.valLoss = valLoss
    print string.format('Val loss: %g', valLoss)

    collectgarbage!
