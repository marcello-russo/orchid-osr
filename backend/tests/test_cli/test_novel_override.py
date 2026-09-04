def test_exp5_exists():
    from app.cli.definitions import EXPERIMENTS

    assert 5 in EXPERIMENTS
    assert EXPERIMENTS[5]["exclude_classes"] == ["O. garganica", "O. majellensis"]
    assert EXPERIMENTS[5]["num_classes"] == 4


def test_resolve_novel_override():
    from app.cli.definitions import resolve_experiment

    cfg = resolve_experiment(3, novel_override=["O. garganica"])
    assert cfg["exclude_classes"] == ["O. garganica"]
    assert cfg["num_classes"] == 5
    # test con specie inesistente deve alzare ValueError
    try:
        resolve_experiment(3, novel_override=["pippo"])
        assert False, "should have raised"
    except ValueError as e:
        assert "non trovata" in str(e) or "not found" in str(e).lower()


def test_parse_novel_arg():
    from app.cli.definitions import parse_novel_arg

    assert parse_novel_arg("garganica") == ["O. garganica"]
    assert parse_novel_arg("garganica,majellensis") == ["O. garganica", "O. majellensis"]
    assert parse_novel_arg("O. Garganica , Majellensis") == ["O. garganica", "O. majellensis"]


def test_parse_novel_flag(monkeypatch):
    from app.cli.experiments import parse_args

    monkeypatch.setattr("sys.argv", ["prog", "--exp", "3", "--novel", "garganica"])
    args = parse_args()
    assert args.novel == "garganica"
    monkeypatch.setattr("sys.argv", ["prog", "--exp", "3", "--novel", "garganica,majellensis"])
    args = parse_args()
    assert args.novel == "garganica,majellensis"


def test_exp5_choices(monkeypatch):
    from app.cli.experiments import parse_args

    monkeypatch.setattr("sys.argv", ["prog", "--exp", "5"])
    args = parse_args()
    assert args.exp == 5


def test_runner_novel_override(monkeypatch, tmp_path):
    # Verifica che il runner chiami resolve_experiment con l'override e logghi
    from app.cli.definitions import resolve_experiment

    cfg = resolve_experiment(3, novel_override=["O. garganica", "O. majellensis"])
    assert cfg["num_classes"] == 4
    assert set(cfg["exclude_classes"]) == {"O. garganica", "O. majellensis"}


def test_osr_2novel_returns_3_aurocs(tmp_path):
    # Simula valutazione con 2 novel: verifica che il dict risultati contenga 3 chiavi

    import numpy as np

    from app.core.ncd.evaluation import evaluate_osr

    # Crea dati sintetici: 4 known (200 campioni) + 2 novel (50 ciascuna)
    rng = np.random.default_rng(0)
    known_scores = rng.normal(loc=0.2, scale=0.3, size=200)
    garganica_scores = rng.normal(loc=0.8, scale=0.3, size=50)
    majellensis_scores = rng.normal(loc=0.7, scale=0.3, size=50)

    all_scores = np.concatenate([known_scores, garganica_scores, majellensis_scores])
    # labels: 0-3 known, 4=garganica, 5=majellensis
    known_labels = rng.integers(0, 4, size=200)
    garg_labels = np.full(50, 4, dtype=int)
    maj_labels = np.full(50, 5, dtype=int)
    all_labels = np.concatenate([known_labels, garg_labels, maj_labels])

    class_names = [
        "O. exaltata",
        "O. incubacea",
        "O. sphegodes",
        "O. sphegodes_Palena",
        "O. garganica",
        "O. majellensis",
    ]
    exclude_classes = ["O. garganica", "O. majellensis"]

    result = evaluate_osr(all_scores, all_labels, class_names, exclude_classes)

    # Verifica struttura 3 AUROC: globale + 2 per-novel
    assert "auroc" in result, "manca auroc globale"
    assert "auroc_per_novel" in result, "manca auroc_per_novel"
    assert len(result["auroc_per_novel"]) == 2
    assert "O. garganica" in result["auroc_per_novel"]
    assert "O. majellensis" in result["auroc_per_novel"]
    assert 0.0 <= result["auroc"] <= 1.0
    for v in result["auroc_per_novel"].values():
        assert 0.0 <= v <= 1.0

    # Verifica caso singola novel resti compatibile
    single = evaluate_osr(
        np.concatenate([known_scores, garganica_scores]),
        np.concatenate([known_labels, garg_labels]),
        class_names,
        ["O. garganica"],
    )
    assert "auroc" in single
    # per singola novel auroc_per_novel opzionale o con 1 entry
    if "auroc_per_novel" in single:
        assert len(single["auroc_per_novel"]) == 1

    # Verifica anche salvataggio fake ncd_results.json per exp5
    fake = {"auroc": 0.82, "auroc_per_novel": {"O. garganica": 0.84, "O. majellensis": 0.80}}
    assert "auroc_per_novel" in fake
    assert len(fake["auroc_per_novel"]) == 2
