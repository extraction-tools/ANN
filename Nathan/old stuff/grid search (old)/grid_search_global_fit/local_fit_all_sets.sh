#!/bin/bash

for i in {0..14}
do
    ./local_fit dvcs_xs_newsets_genCFFs.csv $i
done
