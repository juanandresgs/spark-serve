"""Deterministic synthetic reasoning prompts used by the bounded Qwen check."""
import random

REASONING_FIXTURE_SEEDS = (937, 1459)
PRIMARY_PROMPT_SEED = 61000
SECONDARY_PROMPT_SEED = 72000


def make(seed):
    rng = random.Random(seed)
    fixtures = []
    for _ in range(25):
        x, y, z = [rng.randrange(2, 30) for _ in range(3)]
        fixtures.append((
            "arithmetic",
            f"Compute ({x} * {y}) - {z} * ({x} - {y}).",
            x * y - z * (x - y),
        ))
        values = [rng.randrange(-9, 10) for _ in range(9)]
        fixtures.append((
            "code-semantics",
            f"In Python, what integer does sum(x*x for x in {values!r} if x % 3 == 0) evaluate to?",
            sum(x * x for x in values if x % 3 == 0),
        ))
        fixtures.append((
            "state",
            f"A FIFO queue starts as {values!r}. Append 17, remove the first three items, then append -4. What is the sum of the resulting queue?",
            sum((values + [17])[3:] + [-4]),
        ))
        n = rng.randrange(5, 20)
        edges = [(j, j + 1) for j in range(n - 1)] + [
            (j, j + 3) for j in range(n - 3) if rng.random() < 0.5
        ]
        dist = [0] + [999] * (n - 1)
        for j in range(n):
            for u, v in edges:
                if u == j:
                    dist[v] = min(dist[v], dist[u] + 1)
        fixtures.append((
            "graph",
            f"A directed unweighted graph has edges {edges!r}. What is the minimum number of edges in a path from 0 to {n - 1}?",
            dist[-1],
        ))
    return fixtures


def reasoning_fixtures():
    """Return the ordered 200 prompts and expected integer answers."""
    return make(REASONING_FIXTURE_SEEDS[0]) + make(REASONING_FIXTURE_SEEDS[1])


def request_seed(base_seed, index):
    if base_seed not in (PRIMARY_PROMPT_SEED, SECONDARY_PROMPT_SEED):
        raise ValueError("unknown published reasoning prompt-seed set")
    if type(index) is not int or not 0 <= index < 200:
        raise ValueError("reasoning fixture index must be from 0 through 199")
    return base_seed + index
