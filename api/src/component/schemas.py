import enum

from algorithms.ema import EMA
class AvailableConfigs(enum.Enum):
    BorderCheck = "border_check.json"
    Clustering = "clustering.json"
    Cumulative = "cumulative.json"
    EMAPercentile = "ema_percentile.json"
    EMA = "ema.json"
    Filtering = "filtering.json"
    GAN = "gan.json"
    Hampel = "hampel.json"
    IsolationForest = "isolation_forest.json"
    LinearFit = "linear_fit.json"
    MACD = "macd.json"
    PCA = "pca.json"
    RRCF = "rrcf_trees.json"
    TrendClassification = "trend_classification.json"
    Welford = "welford.json"
    