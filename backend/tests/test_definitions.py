def test_exp5_ncd_mode_true():
    from app.cli.definitions import EXPERIMENTS, EXPERIMENTS_ARCHIVED

    assert EXPERIMENTS[5]["ncd_mode"] is True
    assert EXPERIMENTS[5]["num_classes"] == 4
    assert EXPERIMENTS[5]["exclude_classes"] == ["O. garganica", "O. majellensis"]
    assert 10 not in EXPERIMENTS
    assert 10 in EXPERIMENTS_ARCHIVED


def test_exp4_ncd_mode_true():
    from app.cli.definitions import EXPERIMENTS

    assert EXPERIMENTS[4]["ncd_mode"] is True
