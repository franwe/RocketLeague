import numpy as np


def format_predictions(y_model):
    """
    Back to original structure (team_A/B_scoring = 0/1 instead of team_socring=[-1, 1])
    undo: df["team_scoring_within_10sec"] = -df["team_A_scoring_within_10sec"] + df["team_B_scoring_within_10sec"]
    v < 0: team A scoring
    v > 0: team B scoring
    """
    y = pd.DataFrame(y_model)
    y.columns = ["p"]
    y.reset_index(inplace=True, drop=True)

    y["team_A_scoring_within_10sec"] = y["p"].apply(lambda v: abs(max(v, 0)))
    y["team_B_scoring_within_10sec"] = y["p"].apply(lambda v: min(v, 0))
    return y[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]]


def my_log_loss(y_test_model, y_pred_model):
    clip = lambda p: max(min(p, 1 - 10 ** (-15)), 10 ** (-15))
    y_test = format_predictions(y_test_model)
    y_pred = format_predictions(y_pred_model)
    score = 0
    N = len(y_pred)
    for team in ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]:
        y_test_t = y_test[team].apply(lambda p: clip(p))
        y_pred_t = y_pred[team].apply(lambda p: clip(p))
        team_score = 1 / N * sum(y_test_t * np.log(y_pred_t) + (1 - y_test_t) * np.log(1 - y_pred_t))
        score += team_score
    return -1 / 2 * score
