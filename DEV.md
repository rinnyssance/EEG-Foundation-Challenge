## Hi Folks!

To organize this, I have taken the liberty to modularize our current efforts 
and split the work into the following:

* Data processing
  - This will contain all necessary code for the
    cleaning and preparation of the data for the
    model training.

  - Ideally code from this region should be such when run,
    should prepare the datasets for the other module to use
    for training.

* Model development/construction
  - Code in this section will take the preprocessed data, and
    is solely responsible for just building/experimenting with
    the different model architectures.

  - The intended behaviour of this code in the section here is
    to develop models, run them, and evaluate them based on their
    performance.


## What does this mean?

I think in the end we should have to runnable scripts.
  - One to run for preparing the data,
  - Another to run that will build and save the model.

Meaning the flow of work here is:
  --------------------     ----------------     --------------------------
  |Data Preprocessing| --> |Model training| --> |Saving the model weights|
  --------------------     ----------------     --------------------------

The purpose of this so that different tasks can be worked on concurrently,
and as efficiently as possible. 

> This can always change to fit the needs of the team. For now, this is an
attempt to orgnize the project so that we can progress relatively quickly.
