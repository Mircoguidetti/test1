from costlayer.domains import DOMAINS, load_tasks, make_split


def test_split_is_deterministic_disjoint_and_half_half():
    a, b = make_split(), make_split()
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    for d in DOMAINS:
        n = len(load_tasks(d))
        tr = [r for r in a["train"] if r.domain == d]
        te = [r for r in a["test"] if r.domain == d]
        assert len(tr) + len(te) == n and abs(len(tr) - len(te)) <= 1
    assert len(load_tasks("retail")) == 115 and len(load_tasks("airline")) == 50
