# KrylovPLS
An implementation of the regularised Krylov subspace formulation of partial least squares regression (PLS-R).

# Usage

```python
import PLSR

X, y = load_my_data()

model = PLSR.PLSR(K=2,
                  max_iter=10000,
                  tol=5e-6,
                  solver="krylov",
                  penalty="l1l2",
                  gamma=0.004,
                  alpha=0.9,
                  step_size=0.01,
                  reconstruct_components=True,
                  reconstruct_retries=3,
                  reconstruct_step_size=0.0001,
                  reconstruct_max_iter=10000,
                  reconstruct_tol=5e-10,
                  reconstruct_max_iter_proj=10000,
                  reconstruct_lambda_w_eq_Wk=100.0,
                  reconstruct_lambda_t_eq_Tk=10.0,
                  reconstruct_lambda_beta=1000.0,
                  verbose=1,
                  random_state=42)

model.fit(X, y)

yhat = model.predict(X)
```
# Requirements
Requires "recent" versions of Numpy and scikit-learn (tested with numpy 1.26.4 and scikit-learn 1.2.2).
