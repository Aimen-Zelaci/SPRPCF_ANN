# SPR based PCF sensor ANN

- Train the wgan: `python main.py -train_wgan`

- Generate data: `python main.py -generate`

- Train the ANN model: `python main.py -train_ann`

- Test the ANN model: `python main.py -test_ann`

- PS: gen_iterations is the number of iterations of data generation after training the wgan. if gen_iterations=10 ==> generator will generate 10 * 8 = 80 samples (if all samples pass the filter)
