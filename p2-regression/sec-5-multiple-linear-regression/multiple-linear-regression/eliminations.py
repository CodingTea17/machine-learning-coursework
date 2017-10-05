# REQUIRED TO FIT ORDINARY LEAST SQUARES
import statsmodels.formula.api as sm

def backwards_elimination(x,y,columns,p_sig=0.05):
    features = [i for i in range(0, columns + 1)]
    p_max = 1
    while True:
        x_opt = x[:, features]
        p = (sm.OLS(endog = y, exog = x_opt).fit()).pvalues
        p_max = max(p)
        if p_max > p_sig:
            features.pop(p.tolist().index(p_max))
        else:
            x_opt = x[:, features]
            break;
    return sm.OLS(endog = y, exog = x_opt).fit()
