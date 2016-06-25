VALBATCHES = 100

(model, workers, opts, state) ->
  {:prepBatch, :crit} = state

  ->
    model\evaluate!

    saveseed = torch.random!
    math.randomseed(1234)
    torch.manualSeed(1234)

    valLoss = 0

    for i=1,VALBATCHES
      workers\addjob (-> dataLoader\makebatch 'val'),
        (...) ->
          input, target = prepBatch(...)
          model\forward(input)
          valLoss += crit\forward(model.output, target)

    workers\synchronize!

    math.randomseed(saveseed)
    torch.manualSeed(saveseed)

    valLoss /= VALBATCHES

    state.valLoss = valLoss
    print string.format('Val loss: %g', valLoss)

    collectgarbage!
