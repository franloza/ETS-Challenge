"""ETS Asset Management Factory Challenge 2017

This solution for the challenge is based on the solution provided by
Paul Duan to the Amazon Employee Access Challenge. The repository can
be found in https://github.com/pyduan/amazonaccess/.See README.md for
more details.

Author: Fran Lozano <fjlozanos@gmail.com>
"""

from __future__ import division

import logging,argparse

from sklearn import metrics, model_selection, linear_model, ensemble, neural_network

from helpers import diagnostics
from helpers import ml
from helpers.data import load_data, save_results
from helpers.feature_extraction import create_datasets

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)


def main(CONFIG):
    """
    The final model is a combination of several base models, which are then
    combined using StackedClassifier defined in the helpers.ml module.

    The list of models and associated datasets is generated automatically
    from their identifying strings. The format is as follows:
    A:b_c where A is the initials of the algorithm to use, b is the base
    dataset, and c is the feature set and the variants to use.
    """
    SEED = 42
    selected_models = [
        "GBC:expanded",
    ]

    # Create the models on the fly
    models = []
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = {'LR': linear_model.LogisticRegression,
                 'GBC': ensemble.GradientBoostingClassifier,
                 'RFC': ensemble.RandomForestClassifier,
                 'MLP': neural_network.MLPClassifier,
                 'ETC': ensemble.ExtraTreesClassifier}[model_id]()
        model.set_params(random_state=SEED)
        models.append((model, dataset))

    #datasets = [dataset for model, dataset in models]
    datasets = ["basic", "residuals", "stats", "expanded"]

    logger.info("loading data")
    y, x = load_data('train.csv')
    x_test = load_data('test.csv', return_labels=False)

    logger.info("preparing datasets (use_cache=%s)", str(CONFIG.use_cache))
    create_datasets(x, x_test, y, datasets, CONFIG.use_cache)

    # Set params
    for model, feature_set in models:
        model.set_params(**ml.find_params(model, feature_set, y,
                                          grid_search=CONFIG.grid_search))
    clf = ml.StackedClassifier(
        models, stack=CONFIG.stack, fwls=CONFIG.fwls,
        model_selection=CONFIG.model_selection,
        use_cached_models=CONFIG.use_cache)

    # Results
    # Basic dataset
    #GBC:basic - 5 it: 0.54569
    #LR:basic - 5 it: 0.49412

    #Series dataset
    #GBC:stats - 5 it: 0.76338
    #MLP:stats - 5 it: 0.62772
    #GBC:stats" + RFC:stats - 5 it: 0.73487

    #Expanded dataset
    # GBC:expanded - 5 it: 0.88390
    # RFC:expanded - 5 it: 0.87114
    # ETC:expanded - 5 it: 0.86704
    # GBC + RFC:expanded - 5 it: 0.87741

    #  Metrics
    logger.info("computing cv score")
    mean_auc = 0.0
    for i in range(CONFIG.iter):
        train, cv = model_selection.train_test_split(
            range(len(y)), test_size=.20, random_state=1 + i * SEED)

        cv_preds = clf.fit_predict(y, train, cv, show_steps=CONFIG.verbose)

        fpr, tpr, _ = metrics.roc_curve(y[cv], cv_preds)
        roc_auc = metrics.auc(fpr, tpr)
        logger.info("AUC (it %d/%d): %.5f", i + 1, CONFIG.iter, roc_auc)
        mean_auc += roc_auc

        if CONFIG.diagnostics and i == 0:  # only plot for first it
            logger.info("plotting learning curve")
            clf.use_cached_models = False
            diagnostics.learning_curve(clf, y, train, cv, n=10)
            clf.use_cached_models = True
            diagnostics.plot_roc(fpr, tpr)
    if CONFIG.iter:
        logger.info("Mean AUC: %.5f",  mean_auc/CONFIG.iter)

    # Create submissions
    if CONFIG.outputfile:
        logger.info("making test submissions (CV AUC: %.4f)", mean_auc)
        preds = clf.fit_predict(y, show_steps=CONFIG.verbose)
        save_results(preds, CONFIG.outputfile + ".csv")

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Parameters for the script.")
    PARSER.add_argument('-d', "--diagnostics", action="store_true",
                        help="Compute diagnostics.")
    PARSER.add_argument('-i', "--iter", type=int, default=1,
                        help="Number of iterations for averaging.")
    PARSER.add_argument("-f", "--outputfile", default="",
                        help="Name of the file where predictions are saved.")
    PARSER.add_argument('-g', "--grid-search", action="store_true",
                        help="Use grid search to find best parameters.")
    PARSER.add_argument('-m', "--model-selection", action="store_true",
                        default=False, help="Use model selection.")
    PARSER.add_argument('-n', "--no-cache", action="store_false", default=True,
                        help="Use cache.", dest="use_cache")
    PARSER.add_argument("-s", "--stack", action="store_true",
                        help="Use stacking.")
    PARSER.add_argument('-v', "--verbose", action="store_true",
                        help="Show computation steps.")
    PARSER.add_argument("-w", "--fwls", action="store_true",
                        help="Use metafeatures.")
    PARSER.set_defaults(argument_default=False)
    CONFIG = PARSER.parse_args()

    CONFIG.stack = CONFIG.stack or CONFIG.fwls

    logger.debug('\n' + '='*50)
    main(CONFIG)
