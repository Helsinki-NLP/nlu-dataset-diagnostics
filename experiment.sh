for task in cola sst2 qnli rte qqp stsb mnli mrpc
do
  for dataset in train dev train_dev
  do
    for pos in ADJ ADV CONJ DET NOUN NUM PRON VERB
    do
      bash train.sh $task $dataset $pos
    done
  done
done

