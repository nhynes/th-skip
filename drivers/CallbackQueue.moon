require 'torch'
_ = require 'moses'

CallbackQueue = torch.class('CallbackQueue')

CallbackQueue.__init = (startIter=1) =>
  @q = {}
  @iter = startIter
  @oneshots = 0

CallbackQueue.add = (cb) =>
  assert type(cb.cb) == 'function'

  cb.priority = cb.priority or -math.huge

  if cb.iter and cb.iter < @iter
    return

  if cb.iter
    sortByIter = (cb1, cb2) ->
      cb1.iter == cb2.iter and cb1.priority > cb2.priority or cb1.iter < cb2.iter
    insertInd = _.sortedIndex @q, cb, sortByIter
    table.insert @q, insertInd, cb

    @oneshots += 1 if cb.iterval == nil

  elseif cb.interval
    cb.iter = 1
    table.insert @q, 1, cb

CallbackQueue.waitTime = =>
  #@q == 0 and math.huge or math.max(@q[1].iter - @iter + 1, 0)

CallbackQueue.advance = (nIters) =>
  @iter += nIters or (#@q > 0 and self\waitTime! or 0)

CallbackQueue.pull = => ->
  if @q[1] and @q[1].iter < @iter then
    event = table.remove @q, 1
    if event.interval
      reiter = _.template({
        cb: event.cb,
        iter: @iter + event.interval-1,
        interval: event.interval
      }, event)
      self\add reiter
    else
      @oneshots -= 1

    event.cb

CallbackQueue.__len = => @oneshots
