"""Code from https://github.com/albanie/collaborative-experts/blob/master/model/metric.py """
"""Module for computing performance metrics

"""
import numpy as np
import scipy.stats
import pickle

def t2v_metrics(sims, gt_idx=None, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    
    queries_per_video = num_queries // num_vids

    if gt_idx is None:
        gt_idx = [
            [
                np.ravel_multi_index([ii, jj], (num_queries, num_vids))
                for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)
            ]
            for jj in range(num_vids)
        ]
        gt_idx = np.array(gt_idx)
        # Save gt_idx if a path is provided
        save_gt_idx_path = 'msrvtt/gt_idx.pkl'
        with open(save_gt_idx_path, 'wb') as f:
            pickle.dump(gt_idx, f)
        print(f"gt_idx saved to {save_gt_idx_path}")
    # else:
    #     # gt_idx = np.array(gt_idx, dtype=np.int64)
    #     expected_shape = (num_vids, queries_per_video)
    #     # Validate gt_idx shape
    #     assert gt_idx.shape == expected_shape, f"gt_idx shape should be {expected_shape}"

    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    gt_mean = np.mean(gt_dists)
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT
    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin, 
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    # break_ties = "optimistically"
    break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb; ipdb.set_trace()
    assert cols.size == num_queries, msg

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()
    return cols2metrics(cols, num_queries, gt_mean)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    # switch axes of text and video
    sims = sims.T

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            # Set missing queries to have a distance of infinity.  A missing query
            # refers to a query position `n` for a video that had less than `n`
            # captions (for example, a few MSRVTT videos only have 19 queries)
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                # skip rankings of missing captions
                continue
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    return cols2metrics(query_ranks, num_queries)


def cols2metrics(cols, num_queries,gt_mean):
    metrics = {}
    # metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    # metrics["R2"] = 100 * float(np.sum(cols < 2)) / num_queries
    # metrics["R4"] = 100 * float(np.sum(cols < 4)) / num_queries
    # metrics["R8"] = 100 * float(np.sum(cols < 8)) / num_queries
    # stats = [metrics[x] for x in ("R1", "R2", "R4","R8")]

    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R3"] = 100 * float(np.sum(cols < 3)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    stats = [metrics[x] for x in ("R1", "R5", "R10")]

    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    metrics["MeanA"] = -gt_mean
    for metric in metrics:
        if metric == "MeanA":
            metrics[metric] = round(metrics[metric], 3)
        else:
            metrics[metric] = round(metrics[metric], 1)
    return metrics
