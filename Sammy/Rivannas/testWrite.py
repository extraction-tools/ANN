learning_rate = 0.001
activationFunction = "ReLU"
f = open("/home/sl8rn/test.txt", "x")
f.write("Learning Rate: " + str(learning_rate)+ "\n")
f.write("ActivationFunction: " + activationFunction)
f.close()
