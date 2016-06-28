require 'paths'
_ = require 'moses'

DataLoader = torch.class('DataLoader')

UNK, SOR, EOR, SOS, EOS = 1, 2, 3, 4, 5

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

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = #indsByLen[lengths[i]] for i=1,#lengths -- freq of each index -> len

  with data
    .lengths = lengths
    .indsByLen = {len,torch.LongTensor(inds) for len,inds in pairs indsByLen}
    .lenFreqs = lenFreqs

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
  batchSents\select(2, 1)\fill(SOS)
  batchSents\select(2, sentlen+2)\fill(EOS)

  batchSents

DataLoader.partitionSize = (partition='train') =>
  ds = partition == 'train' and @dataTrain or @dataVal
  ds.toks\size(1)

DataLoader.terminate = =>
