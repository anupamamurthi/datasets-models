import os
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.models import model_from_json
from os.path import abspath
import sys
sys.path.append(abspath('./adversarial-robustness-toolbox'))


def main(argv):
    if len(argv) < 2:
        sys.exit("Not enough arguments provided.")

    global network_definition_filename, weights_filename, dataset_filename

    i = 1
    while i <= 6:
        arg = str(argv[i])
        print(arg)
        if arg == "--data":
            dataset_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i+1]))
        if arg == "--networkdefinition":
            network_definition_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i+1]))
        if arg == "--weights":
            weights_filename = os.path.join(os.environ["DATA_DIR"], str(argv[i+1]))

        i += 2

    print(dataset_filename)
    print(network_definition_filename)
    print(weights_filename)

    json_file = open(network_definition_filename, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weights_filename)
    comp_params = {'loss': 'categorical_crossentropy',
                   'optimizer': 'adam',
                   'metrics': ['accuracy']}
    model.compile(**comp_params)



    from art.classifiers.keras import KerasClassifier

    classifier = KerasClassifier((0, 1), model)

    from keras.utils import np_utils

    pf = np.load(dataset_filename)

    x = pf['x_test']
    y = pf['y_test']

    x = np.expand_dims(x, axis=3)
    x = x.astype('float32') / 255

    y = np_utils.to_categorical(y, 10)

    # evaluate
    scores = model.evaluate(x, y, verbose=0)
    print('model test loss : ', scores[0] * 100)
    print('model test accuracy : ', scores[1] * 100)
    model_accuracy = scores[1] * 100

    from art.attacks.fast_gradient import FastGradientMethod
    epsilon = 0.2

    # check if we want to craft samples or not
    crafter = FastGradientMethod(classifier, eps=epsilon)

    x_samples = crafter.generate(x)
    print(x_samples.shape)

    scores = model.evaluate(x, y, verbose=0)
    model_accuracy = scores[1] * 100

    y_pred = model.predict(x, verbose=0)
    y_pred_adv = model.predict(x_samples, verbose=0)

    # evaluate
    scores = model.evaluate(x_samples, y, verbose=0)
    print(scores)
    # # obtain results/metrics
    # pert_metric = self.perturb_metric(x, x_samples, y_pred, y_pred_adv, ord=2)
    # cmetric = self.conf_metric(x, x_samples, y_pred, y_pred_adv, ord=2)
    # reduction_in_confidence = cmetric * 100

    # data = {
    #     "model accuracy on test data:": model_accuracy,
    #     "model accuracy on adversarial samples": scores[1] * 100,
    #     "reduction in confidence": reduction_in_confidence,
    #     "average perturbation": pert_metric * 100
    # }

    # print(json.dumps(data, indent=4, sort_keys=True))

if __name__ == "__main__":
    main(sys.argv)


