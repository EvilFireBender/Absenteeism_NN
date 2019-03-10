# Absenteeism_NN

Feed forward networks that train using back-propagation have been
a much discussed subject in industry and research. Such networks facilitate
systems that given the right type (and size) of data, predict results or classify
objects with relative ease.

One common complaint with such systems has been the following - there
seems to be no reliable way of knowing in advance, the number of hidden layer
neurons needed for optimum functioning of the network. And often,
satisfactory results are obtained after overestimating this number, in which
case, the system can be made more efficient if these neurons are pruned. This
paper studies the effects of one such pruning strategy: pruning by Badness
factor.

Our network predicts absenteeism at work using factors such as employee
weight, height, BMI, distance from workplace, prior record of social
smoking/drinking and so on. We do this by training our network on a record of
employee absenteeism from a courier company in Brazil. The ANN classifies
the absenteeism into one of three categories, and we assess the accuracy of the
system using various means. Then we prune one of the excessive nodes, and
re-evaluate our system’s accuracy. While accuracy is seen to marginally
improve in 60% of the cases, it stays the same in 10% of the runs and decreases
in the remaining 30%. Pruning by badness is found to be a not too reliable
strategy for reducing extra neurons. The paper also discusses the application of
Deep Neural Networks to the same problem, although the implementation of it ends
up not working.
