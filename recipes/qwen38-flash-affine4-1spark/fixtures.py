import random
def make(seed=147):
 rng=random.Random(seed);fixtures=[]
 for i in range(25):
  x,y,z=[rng.randrange(2,30) for _ in range(3)]
  fixtures.append(('arithmetic',f'Compute ({x} * {y}) - {z} * ({x} - {y}).',x*y-z*(x-y)))
  values=[rng.randrange(-9,10) for _ in range(9)]
  fixtures.append(('code-semantics',f'In Python, what integer does sum(x*x for x in {values!r} if x % 3 == 0) evaluate to?',sum(x*x for x in values if x%3==0)))
  fixtures.append(('state',f'A FIFO queue starts as {values!r}. Append 17, remove the first three items, then append -4. What is the sum of the resulting queue?',sum((values+[17])[3:]+[-4])))
  n=rng.randrange(5,20);edges=[(j,j+1) for j in range(n-1)]+[(j,j+3) for j in range(n-3) if rng.random()<.5]
  dist=[0]+[999]*(n-1)
  for j in range(n):
   for u,v in edges:
    if u==j:dist[v]=min(dist[v],dist[u]+1)
  fixtures.append(('graph',f'A directed unweighted graph has edges {edges!r}. What is the minimum number of edges in a path from 0 to {n-1}?',dist[-1]))
 return fixtures
fixtures=make()
