"""🧑‍💻 command line interface for NiaAML"""

from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import pandas as pd
import typer

from niaaml import PipelineOptimizer, Pipeline
from niaaml.data.csv_data_reader import CSVDataReader


app = typer.Typer(
    help="🌳 a command line interface for NiaAML.",
    no_args_is_help=True
)


@app.command()
def optimize(
    data_csv_file: Path,
    has_header: bool = True,
    ignore_columns: list[int] = [],
    classifiers: list[str] = ['AdaBoost', 'Bagging', 'MultiLayerPerceptron', 'RandomForest', 'ExtremelyRandomizedTrees', 'LinearSVC'],
    feature_selection_algorithms: list[str] = ['SelectKBest', 'SelectPercentile', 'ParticleSwarmOptimization', 'VarianceThreshold'],
    feature_transform_algorithms: list[str] = ['Normalizer', 'StandardScaler'],
    categorical_features_encoder: Annotated[Optional[str], typer.Option()] = "OneHotEncoder",
    imputer: Annotated[Optional[str], typer.Option()] = None,
    fitness_name: str = 'Accuracy',
    pipeline_population_size: int = 15,
    inner_population_size: int = 15,
    number_of_pipeline_evaluations: int = 100,
    number_of_inner_evaluations: int = 100,
    optimization_algorithm: str = 'ParticleSwarmAlgorithm',
    inner_optimization_algorithm: Annotated[Optional[str], typer.Option()] = None,
    result_file: Path = Path("pipeline.ppln"),
) -> None:
    """🦾 optimizes a NiaAML pipeline on a given dataset."""
    # 📄 load and setup data
    logger.info(f"📄 reading `{data_csv_file}`")
    data_reader = CSVDataReader(
        src=str(data_csv_file),
        has_header=has_header,
        contains_classes=True,
        ignore_columns=ignore_columns
    )

    # 🦾 setup pipeline
    logger.info("🦾 start the optimization process ...")
    pipeline_optimizer = PipelineOptimizer(
        data=data_reader,
        classifiers=classifiers,
        feature_selection_algorithms=feature_selection_algorithms,
        feature_transform_algorithms=feature_transform_algorithms,
        categorical_features_encoder=categorical_features_encoder,
        imputer=imputer,
    )

    # 📈 optimize pipeline
    pipeline = pipeline_optimizer.run(fitness_name, pipeline_population_size, inner_population_size, number_of_pipeline_evaluations, number_of_inner_evaluations, optimization_algorithm, inner_optimization_algorithm)

    # 💾 save pipeline
    logger.success(f"💾 saving optimized pipeline to `{result_file}`")
    pipeline.export(result_file)

@app.command()
def infer(
    data_csv_file: Path,
    has_header: bool = True,
    ignore_columns: list[int] = [],
    pipeline_file: Path = Path("pipeline.ppln"),
    predictions_csv_file: Path = Path("preds.csv"),
) -> None:
    """🔮 use an optimized NiaAML pipeline to make predictions."""
    # 💾 load pipeline
    pipeline = Pipeline.load(pipeline_file)

    # 📄 load and setup data
    logger.info(f"📄 reading `{data_csv_file}`")
    reader = CSVDataReader(
        src=str(data_csv_file),
        has_header=has_header,
        contains_classes=True,
        ignore_columns=ignore_columns
    )
    reader._read_data()
    x: pd.DataFrame = reader._x

    # 🔮 make predictions
    logger.info(f"🔮 using `{pipeline_file}` to make predictions on the data")
    x['preds'] = pipeline.run(x)

    # 💾 save predictions
    logger.success(f"💾 saving predictions to `{predictions_csv_file}`")
    x.to_csv(predictions_csv_file)

def main():
    """🚪 typer entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
