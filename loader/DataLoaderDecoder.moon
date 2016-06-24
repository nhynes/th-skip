require 'paths'
_ = require 'moses'

DataLoader = torch.class('DataLoader')

UNK, SOR, EOR, EOS = 1, 2, 3, 4

groupByLen = (data) ->
  slens = data.slens

  indsByLen = {}
  for i=1,slens\size(1)
    slen = slens[i]

    ibrl = indsByLen[slen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[slen] = ibrl

  lengths = _.keys indsByLen
  table.sort lengths

  if lengths[1] ~= 1
    indsByLen[1] = {}
    table.insert(lengths, 0, 1) if lengths[1] ~= 1

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = #indsByLen[lengths[i]] for i=1,#lengths -- freq of each index -> len

  maxFreq = lenFreqs\max!
  soreorFreq = 1 - lenFreqs[1]/maxFreq
  lenFreqs[1] = maxFreq  -- 1 - soreorFreq/lenFreqs of the time, present <[se]or>

  with data
    .lengths = lengths
    .indsByLen = {len,torch.LongTensor(inds) for len,inds in pairs indsByLen}
    .lenFreqs = lenFreqs
    .soreorFreq = soreorFreq

DataLoader.__init = (dataTrain, dataVal, opts) =>
  @batchSize = opts.batchSize
  @dataTrain = dataTrain
  @dataVal = dataVal

  @vocabSize = opts.vocabSize

  groupByLen(dataTrain)
  groupByLen(dataVal)

DataLoader.makebatch = (partition='train') =>
  data = partition == 'val' and @dataVal or @dataTrain
  toks = data.toks

  sentlen = data.lengths[torch.multinomial(data.lenFreqs, 1)[1]]

  if sentlen == 1 and math.random() < data.soreorFreq -- train on <r> </r>
    batchSents = torch.LongTensor(@batchSize, 3)\fill(EOS)
    batchSents\select(2, 2)\random(2, 3)
    return batchSents


  sentlenInds = data.indsByLen[sentlen]
  batchSize = math.min(sentlenInds\size(1), @batchSize)

  selIndInds = torch.randperm(sentlenInds\size(1))\sub(1, batchSize)\long!
  selInds = sentlenInds\index(1, selIndInds)

  strides = torch.LongStorage{sentlen+2, 1}
  batchSents = toks.new(batchSize, sentlen+2)\zero!
  batchSentsIdx = with toks.new!
    \set batchSents\storage!, 2,
      torch.LongStorage {batchSize, toks\size(2)},
      strides
  batchSentsIdx\index(toks, 1, selInds)
  batchSents\select(2, 1)\fill(EOS)
  batchSents\select(2, sentlen+2)\fill(EOS)

  batchSents

DataLoader.partitionSize = (partition='train') =>
  ds = partition == 'train' and @dataTrain or @dataVal
  ds.toks\size(1)

DataLoader.terminate = =>
