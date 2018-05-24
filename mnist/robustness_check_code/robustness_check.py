def get_metrics(model, x_original, x_adv_samples, y):
    scores = model.evaluate(x_original, y, verbose=0)
    model_accuracy_on_non_adversarial_samples = scores[1] * 100

    y_pred = model.predict(x_original, verbose=0)
    y_pred_adv = model.predict(x_adv_samples, verbose=0)

    scores = model.evaluate(x_adv_samples, y, verbose=0)
    model_accuracy_on_adversarial_samples = scores[1] * 100

    pert_metric = get_perturbation_metric(x_original, x_adv_samples, y_pred, y_pred_adv, ord=2)
    conf_metric = get_confidence_metric(y_pred, y_pred_adv)

    data = {
        "model accuracy on test data:": model_accuracy_on_non_adversarial_samples,
        "model accuracy on adversarial samples": model_accuracy_on_adversarial_samples,
        "reduction in confidence": conf_metric * 100,
        "average perturbation": pert_metric * 100
    }
    return data


def get_perturbation_metric(x_original, x_adv, y_pred, y_pred_adv, ord=2):

    idxs = (np.argmax(y_pred_adv, axis=1) != np.argmax(y_pred, axis=1))

    if np.sum(idxs) == 0.0:
        return 0

    perts_norm = la.norm((x_adv - x_original).reshape(x_original.shape[0], -1), ord, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x_original[idxs].reshape(np.sum(idxs), -1), ord, axis=1))


# This computes the change in confidence for all images in the test set
def get_confidence_metric(y_pred, y_pred_adv):

    y_classidx = np.argmax(y_pred, axis=1)
    y_classconf = y_pred[np.arange(y_pred.shape[0]), y_classidx]

    y_adv_classidx = np.argmax(y_pred_adv, axis=1)
    y_adv_classconf = y_pred_adv[np.arange(y_pred_adv.shape[0]), y_adv_classidx]

    idxs = (y_classidx == y_adv_classidx)

    if np.sum(idxs) == 0.0:
        return 0

    idxnonzero = y_classconf != 0
    idxs = idxs & idxnonzero

    return np.mean((y_classconf[idxs] - y_adv_classconf[idxs]) / y_classconf[idxs])
