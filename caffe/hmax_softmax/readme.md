This model provides a softmax classifier acting as the conceptual layer on top
of a traditional HMAX model with universal features developed by Max
Riesenhuber and Josh Rule <rsj28@georgetown.edu>. The classifier takes as input
vectors of 3,200 universal features, passes them through a fully connected
layer to produce an output vector of 2,000 features (one for each of 2,000
vocabulary concepts). These outputs are then passed through a softmax function
to produce a final judgment. 
