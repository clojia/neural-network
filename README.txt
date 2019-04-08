############ files ###########
- run.py
main function: select experiments: "testIdentity", "testTennis", "testIris" and "testIrisNoisy".

- preprocessor.py
Preprocessor class: load data, encode categorical data, convert data to matrix for later training

- network.py
NNUtil and NeuralNetwork class: model training (backpropagation), predicting and calculating accuracy 

- /data/identity/*.txt
identiy data files for experiment "testIdentity"

- /data/tennis/*.txt
tennis data files for experiment "testTennis"

- /data/iris/*.txt
iris data files for experiments "testIris" and "testIrisNoisy"


############# Usage ###########
python3 run.py -e <experiment> (need install numpy)

e.g.
"""
python3 run.py -e testTennis
"""
