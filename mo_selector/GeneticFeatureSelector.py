import multiprocessing
import os
import random
from functools import partial
from time import time
from typing import List, Callable, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from typeguard import typechecked


@typechecked
class GeneticFeatureSelector:

    def __init__(self,
                 model: BaseEstimator,
                 x_train: pd.DataFrame,
                 y_train: pd.Series,
                 all_features: List[str],
                 evaluation_metrics: List[Callable[..., Any]],
                 cv_evaluation_metric_index: int,
                 metrics_weights: List[float],
                 categorical_features: List[str] = None,
                 default_encoder: any = OneHotEncoder(categories='auto', handle_unknown='ignore'),
                 default_scaler: any = StandardScaler(),
                 sampling_size: int = None,
                 cv: int = 10,
                 minimize_num_of_features: bool = True,
                 enable_cv_shuffle_split: bool = False,
                 enable_train_test_split_shuffle: bool = False,
                 shuffle_training_data_every_generation: bool = False,
                 cross_validation_in_objective_func: bool = False,
                 objective_func_cv: int = 3,
                 x_val: Optional[pd.DataFrame] = None,
                 y_val: Optional[pd.Series] = None,
                 n_jobs: int = 1,
                 verbose: bool = True) -> None:
        """
        This class creates a genetic feature selector using multi objective optimization.

        :param model: A sklearn model.
        :param x_train: Training data.
        :param y_train: Training labels.
        :param x_val: Validation data.
        :param y_val: Validation labels.
        :param all_features: A list of all possible features.
        :param evaluation_metrics: A list of evaluation metrics to use for the optimization.
        :param cv_evaluation_metric_index: The index of the evaluation metric to use for the cross validation.
        :param metrics_weights: A list of weights for the evaluation metrics.
        :param categorical_features: A list of the categorical features.
        :param default_encoder: A sklearn transformer to use for categorical features.
        :param default_scaler: A sklearn transformer to use for numerical features.
        :param sampling_size: The size of the sample to use for optimization.
        :param cv: The number of folds to use in the cross validation.
        :param minimize_num_of_features: If True, the genetic algorithm will minimize the number of features.
        :param enable_cv_shuffle_split: If True, the cross validation will use the ShuffleSplit method.
        :param enable_train_test_split_shuffle: If True, the train_test_split will use the ShuffleSplit method.
        :param shuffle_training_data_every_generation: If True, the training data will be shuffled every generation.
        :param cross_validation_in_objective_func: If True, the objective function will use the cross validation.
        :param objective_func_cv: The number of folds to use in the objective function.
        :param n_jobs: The number of jobs to use in the genetic algorithm.
        :param verbose: If True, the genetic algorithm will print the current generation.
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.all_features = all_features
        self.evaluation_metrics = evaluation_metrics
        self.cv_evaluation_metric_index = cv_evaluation_metric_index
        self.metrics_weights = metrics_weights
        if categorical_features is None:
            self.categorical_features = []
        else:
            self.categorical_features = categorical_features
        self.default_encoder = default_encoder
        self.default_scaler = default_scaler
        self.sampling_size = sampling_size
        self.cv = cv
        self.minimize_num_of_features = minimize_num_of_features
        self.enable_cv_shuffle_split = enable_cv_shuffle_split
        self.enable_train_test_split_shuffle = enable_train_test_split_shuffle
        self.shuffle_training_data_every_generation = shuffle_training_data_every_generation
        self.cross_validation_in_objective_func = cross_validation_in_objective_func
        self.objective_func_cv = objective_func_cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.ga_solution_length = len(all_features)
        self.best_solution: Optional[List[Union[int, float]]] = None
        self.best_score: Optional[float] = None
        self.final_pipeline: Optional[Pipeline] = None

        self.metrics_names = [metric.__name__ for metric in self.evaluation_metrics]

        if self.minimize_num_of_features:
            self.metrics_weights.append(-1.0)
            self.metrics_names.append('n_features')

        self.metrics_names = ', '.join(self.metrics_names)
        self.metrics_names = '(' + self.metrics_names + ')'

        creator.create("Fitness", base.Fitness, weights=tuple(self.metrics_weights))
        creator.create("Individual", list, fitness=creator.Fitness)

    def _create_optimization_toolbox(self):
        """
        Create deap optimization toolbox and configure NSGA-III to use binary solution.

        :param self: The object pointer.
        :return: The deap optimization toolbox.
        """
        num_of_reference_points = 10
        ref_points = tools.uniform_reference_points(len(self.metrics_weights), num_of_reference_points)

        # Create the toolbox
        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool,
                         n=self.ga_solution_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        if self.shuffle_training_data_every_generation:
            toolbox.register("evaluate", self._objective_function, gen=0)
        else:
            toolbox.register("evaluate", self._objective_function)

        # configure NSGA-III to use binary solution
        toolbox.register("cross_validation", self._cross_validation_function)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        # use multiprocessing
        n_cpus = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        pool = multiprocessing.Pool(n_cpus)
        toolbox.register("map", pool.map)

        return toolbox

    def get_selected_features(self, individual: List[Union[int, float]]) -> List[str]:
        """
        Get the selected features from the individual.

        :param individual: The individual.
        :return: The selected features.
        """
        selected_features = [feature for feature, value in zip(self.all_features, individual) if value > 0.5]
        return selected_features

    def _create_pipeline(self,
                         selected_categorical_features: List[str],
                         selected_categorical_features_length: int,
                         selected_features_length: int) -> Pipeline:
        """
        Create the pipeline using the selected features.
        :param selected_categorical_features: The selected categorical features.
        :param selected_categorical_features_length: The length of the selected categorical features.
        :param selected_features_length: The length of the selected features.
        :return:
        """
        if selected_categorical_features_length > 0:
            if selected_categorical_features_length < selected_features_length:
                # use the default pipeline
                column_transformer = make_column_transformer(
                    (self.default_encoder, selected_categorical_features),
                    remainder=self.default_scaler,
                    n_jobs=1)
            else:
                # if all categorical features are selected, encode them using the default encoder
                column_transformer = make_column_transformer(
                    (self.default_encoder, selected_categorical_features),
                    n_jobs=1)

            pipeline = make_pipeline(column_transformer, self.model)
        else:
            # no categorical features, only use a default scaler
            pipeline = make_pipeline(self.default_scaler, self.model)

        return pipeline

    def _cross_validation_function(self, individual: List[Union[int, float]]) -> float:
        """
        The objective function for cross validation.
        :param individual: The individual.
        :return: The mean score.
        """

        selected_features = self.get_selected_features(individual)
        selected_features_length = len(selected_features)

        evaluation_metric = self.evaluation_metrics[self.cv_evaluation_metric_index]
        evaluation_metric_weight = self.metrics_weights[self.cv_evaluation_metric_index]

        if selected_features_length == 0:
            return evaluation_metric_weight * 1e7

        pipeline = self.create_pipeline_from_features(selected_features)

        selected_x_train = self.x_train.loc[:, selected_features].copy()

        evaluation_metric_scorer = make_scorer(evaluation_metric, greater_is_better=True)
        scores = cross_val_score(estimator=pipeline, X=selected_x_train, y=self.y_train, cv=self.cv,
                                 scoring=evaluation_metric_scorer, n_jobs=-1)
        mean_score = scores.mean()

        return mean_score

    def _objective_function(self, individual: List[Union[int, float]], gen=0) -> Tuple[float, ...]:
        """
        Objective function for the genetic algorithm.

        :param individual: The individual to evaluate.
        :param gen: The generation number.
        :return: The objective function value.
        """
        selected_features = self.get_selected_features(individual)
        selected_features_length = len(selected_features)

        # penalize the solution if no features were selected
        if selected_features_length == 0:
            output = [weight * -1e7 for weight in self.metrics_weights]
            return tuple(output)

        # create the pipeline
        pipeline = self.create_pipeline_from_features(selected_features)

        selected_x_train = self.x_train.loc[:, selected_features].copy()

        if self.cross_validation_in_objective_func:
            y_pred = cross_val_predict(estimator=pipeline,
                                       X=selected_x_train.copy(),
                                       y=self.y_train.copy(),
                                       cv=self.objective_func_cv)

            scores = [evaluation_metric(self.y_train, y_pred) for evaluation_metric in self.evaluation_metrics]
        else:
            opt_x_train, opt_x_test, opt_y_train, opt_y_test = train_test_split(selected_x_train,
                                                                                self.y_train.copy(),
                                                                                test_size=0.3, random_state=gen,
                                                                                shuffle=self.enable_train_test_split_shuffle)

            pipeline.fit(opt_x_train, opt_y_train)
            y_pred = pipeline.predict(opt_x_test)

            scores = [evaluation_metric(opt_y_test, y_pred) for evaluation_metric in self.evaluation_metrics]

        # if the user passed a validation set
        # use it to evaluate the model by merging its scores with the previous scores
        if self.x_val is not None and self.y_val is not None:
            x_val = self.x_val.loc[:, selected_features].copy()
            y_val = self.y_val.copy()
            pipeline.fit(selected_x_train.copy(), self.y_train.copy())
            y_pred = pipeline.predict(x_val)
            all_scores = [[evaluation_metric(y_val, y_pred) for evaluation_metric in self.evaluation_metrics],
                          scores]
            scores = np.array(all_scores).mean(axis=0).tolist()

        if self.minimize_num_of_features:
            scores.append(selected_features_length)

        return tuple(scores)

    def create_pipeline(self, individual: List[Union[int, float]]) -> Pipeline:
        """
        Create ML pipeline from an individual solution.
        :param individual:  The individual to create the pipeline from.
        :return: The created pipeline.
        """
        selected_features = self.get_selected_features(individual)

        return self.create_pipeline_from_features(selected_features)

    def create_pipeline_from_features(self, selected_features: List[str]) -> Pipeline:
        """
        Create ML pipeline from a list of selected features.
        :param selected_features: The list of selected features.
        :return:cThe created pipeline.
        """
        selected_features_length = len(selected_features)

        # Filter the selected categorical features and create the machine learning pipeline
        selected_categorical_features = [feature for feature in selected_features if
                                         feature in self.categorical_features]
        selected_categorical_features_length = len(selected_categorical_features)

        pipeline = self._create_pipeline(selected_categorical_features, selected_categorical_features_length,
                                         selected_features_length)

        return pipeline

    def fit(self,
            number_of_generations: int = 100,
            mu: int = 10,
            crossover_probability: float = 0.5,
            mutation_probability: float = 0.2,
            early_stopping_patience: int = 5,
            random_state: int = 77):

        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        toolbox = self._create_optimization_toolbox()

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats = tools.MultiStatistics(fitness=stats_fit)
        stats.register("mean" + self.metrics_names, np.mean, axis=0)
        # stats.register("std", np.std, axis=0)
        stats.register("min" + self.metrics_names, np.min, axis=0)
        stats.register("max" + self.metrics_names, np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = stats.fields if stats else []

        pop = toolbox.population(n=mu)

        best_validation_solution = None
        best_validation_score = None
        best_validation_equality_counter = 0
        max_best_validation_equality_counter = early_stopping_patience

        if self.verbose:
            print("Start of evolution")

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]

        if self.shuffle_training_data_every_generation:
            fitnesses = toolbox.map(partial(toolbox.evaluate, gen=0), invalid_ind)
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop) if stats else {}
        logbook.record(_gen=0,
                       time_in_seconds=np.nan,
                       _evals=len(invalid_ind),
                       _gen_val_score=np.nan,
                       best_val_score=np.nan,
                       **record)

        if self.verbose:
            print(logbook.stream)

        for gen in range(1, number_of_generations + 1):
            gen_start = time()

            offspring = algorithms.varAnd(pop, toolbox, crossover_probability, mutation_probability)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if self.shuffle_training_data_every_generation:
                fitnesses = toolbox.map(partial(toolbox.evaluate, gen=gen), invalid_ind)
            else:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            current_best_validation_solution = tools.selBest(pop, 1)[0]
            current_best_validation_score = toolbox.cross_validation(current_best_validation_solution)
            minimize_metric = self.metrics_weights[self.cv_evaluation_metric_index] < 0
            if best_validation_solution is not None:
                if (current_best_validation_score < best_validation_score and minimize_metric) or (
                        current_best_validation_score > best_validation_score and not minimize_metric):
                    best_validation_solution = current_best_validation_solution.copy()
                    best_validation_score = current_best_validation_score
                    best_validation_equality_counter = 0
                elif (current_best_validation_score >= best_validation_score and minimize_metric) or (
                        current_best_validation_score <= best_validation_score and not minimize_metric):
                    best_validation_equality_counter += 1

                if best_validation_equality_counter >= max_best_validation_equality_counter:
                    break
            else:
                best_validation_solution = current_best_validation_solution.copy()
                best_validation_score = current_best_validation_score

            # Select the next generation population from parents and offspring
            pop = toolbox.select(pop + offspring, mu)

            gen_stop = time()
            # Append the current generation statistics to the logbook
            record = stats.compile(pop) if stats else {}
            logbook.record(_gen=gen,
                           time_in_seconds=(gen_stop - gen_start),
                           _evals=len(invalid_ind),
                           _gen_val_score=current_best_validation_score,
                           best_val_score=best_validation_score,
                           **record)
            if self.verbose:
                print(logbook.stream)

        self.best_solution = best_validation_solution.copy()
        self.best_score = best_validation_score

        results_data = {
            'features': []
        }

        if self.minimize_num_of_features:
            results_data['n_features'] = []

        for metric in self.evaluation_metrics:
            results_data[metric.__name__] = []

        pareto_front = tools.sortLogNondominated(pop, k=1)[0]
        pareto_front_fit = np.array([ind.fitness.values for ind in pareto_front])

        for solution, metrics in zip(pareto_front, pareto_front_fit):
            selected_features = self.get_selected_features(solution)
            n_features = metrics[-1]
            results_data['features'].append(selected_features)

            for index, metric in enumerate(self.evaluation_metrics):
                results_data[metric.__name__].append(metrics[index])

            if self.minimize_num_of_features:
                results_data['n_features'].append(n_features)

        results_df = pd.DataFrame(results_data)
        results_df['features'] = results_df['features'].astype(str)
        results_df.drop_duplicates().reset_index(drop=True, inplace=True)
        return pareto_front, results_df, self.get_selected_features(self.best_solution)
