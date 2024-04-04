from mbr import MBRConfig
from base import metric_is_source_based, MetricRunner


def load_metric_runner(mbr_config: MBRConfig, tokenizer=None) -> MetricRunner:
    if mbr_config.metric in {"fastchrf", "aggregate_chrf", "fastchrf.aggregate_chrf"}:
        from fastchrf import FastChrfMetricRunner
        return FastChrfMetricRunner(mbr_config, tokenizer, compute_pairwise_average=False)
    elif mbr_config.metric in {"pairwise_chrf", "fastchrf.pairwise_chrf"}:
        from fastchrf import FastChrfMetricRunner
        return FastChrfMetricRunner(mbr_config, tokenizer, compute_pairwise_average=True)
    elif mbr_config.metric == "comet":
        from comet import CometMetricRunner
        return CometMetricRunner(mbr_config, tokenizer,
                                 device=0,
                                 batch_size_embed=64,
                                 batch_size_estimate=64,
                                 progress_bar=True,
                                 )
    elif mbr_config.metric == "aggregate_comet":
        from comet import AggregateCometMetricRunner
        mbr_config.metric = "comet"
        return AggregateCometMetricRunner(mbr_config, tokenizer,
                                          device=0,
                                          batch_size_embed=64,
                                          batch_size_estimate=64,
                                          progress_bar=True,
                                          )
    else:
        return MetricRunner(mbr_config, tokenizer)
