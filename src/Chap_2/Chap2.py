from math import sqrt

import pandas as pd
# import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x)  # pandas
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 600)

# import seaborn as sns
# %matplotlib inline
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
import statsmodels
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import f

"""
rm(list=ls())
setwd('E:/Book/Ch_2')  # replace with own directory

library(data.table)
library(ggplot2)
library(lmtest)
library(strucchange)
library(tseries)
library(xtable)

### 2.8 Simulated Data ###
sim.data < - list()
sim.data$full < - fread('simulated_datac2.csv')
sim.data$est < - sim.data$full[102:301]  # estimation sample
sim.data$fore < - sim.data$full[302:401]  # forecasting sample

"""

sim_data = r'https://storage.googleapis.com/applied-economics/simulated_datac2.csv'

df_sim_data_full = pd.read_csv(sim_data, header=0, index_col=0, parse_dates=True).reset_index()

df_sim_data_est = df_sim_data_full[101:301]
df_sim_data_fore = df_sim_data_full[301:401]

reg = smf.ols('y ~ x', data=df_sim_data_est).fit()

reg.summary()

pred_val = reg.fittedvalues.copy()
true_val = df_sim_data_est['y'].copy()

residual = true_val - pred_val

fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(residual, pred_val)

# testing homoskedasticy

"""
0.14353  Lagrange multiplier statistic
0.70480                        p-value
0.14219                        f-value
0.70652                      f p-value

BP = 0.1372, df = 1, p-value = 0.7111
"""
"""
ols.fit < - lm(y~ x, data = sim.data$est)

"""

ols_fit = smf.ols('y ~ x', data=df_sim_data_est).fit()
df_sim_data_est['eps'] = ols_fit.resid

"""
# Breusch Pagan Test
bptest(ols.fit)

"""

# 1) Breusch-Pagan-Godfrey (BPGT) test

name = ['Lagrange multiplier statistic','p-value','f-value','f p-value']
bptest = statsmodels.stats.diagnostic.het_breuschpagan(reg.resid, reg.model.exog)
pd.DataFrame(name,bptest)

"""
# Normality test & Histogram

sim.data$est[, eps: = ols.fit$residuals]
hist(sim.data$est[, eps], xlab = '', ylab = '', main = '')
jarque.bera.test(sim.data$est[, eps])

data:  sim.data$est[, eps]
X-squared = 2.4727, df = 2, p-value = 0.2904

[('Jarque-Bera', 2.4727215410116363),
 ('Chi^2 two-tail prob.', 0.2904392721478955),
 ('Skew', -0.25148700038845034),
 ('Kurtosis', 2.7908499594456777)]

"""

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(df_sim_data_est['eps'])
lzip(name, test)

"""

# White test
sim.data$est[, c('x2', 'eps2'): = list(x ^ 2, eps ^ 2)]
white.fit < - lm(eps2 ~ x2 + x, data = sim.data$est)
summary(white.fit)

# Durbin-Watson test
dwtest(ols.fit)

Durbin-Watson test

data:  ols.fit
DW = 1.0287, p-value = 2.651e-12
alternative hypothesis: true autocorrelation is greater than 0

Out[40]: 1.0286553284745694

"""

df_sq = pd.concat([df_sim_data_est['x'] **2, df_sim_data_est['eps'] **2], axis=1)
df_sq.columns = ['x2', 'eps2']

df_sim_data_est = df_sim_data_est.join(df_sq)

white_fit = smf.ols('eps2 ~ x2 + x', data=df_sim_data_est).fit()
print(white_fit.summary())

statsmodels.stats.stattools.durbin_watson(ols_fit.resid)


"""
# Chow break point test
sctest(y
~ x, data = sim.data$est, type = 'Chow',
from = 0.1, to = 0.25)
sctest(y
~ x, data = sim.data$est, type = 'Chow',
from = 0.25, to = 0.5)
sctest(y
~ x, data = sim.data$est, type = 'Chow',
from = 0.5, to = 0.75)

fs < - Fstats(y
~ x, data = sim.data$est,
from = 0.25, to = 1)
fs < - Fstats(y
~ x, data = sim.data$est,
from = 0.5, to = 1)
fs < - Fstats(y
~ x, data = sim.data$est,
from = 0.75, to = 1)
"""

# SKIPPED Chow break point test

"""

# Bai-Perron test
breakpoints(y ~ x, data = sim.data$est, h = 0.15)


"""
# SKIPPED Bai-Perron test

"""

# Dummy variable
olsD.fit < - lm(y~ D + x + D * x, data = sim.data$est)
summary(olsD.fit)

"""

# mod = smf.ols("y~ D + x + D * x", data=df_sim_data_est).fit()

olsD_fit = smf.ols("y~ D + x + D * x", data=df_sim_data_est).fit()
print(olsD_fit.summary())

"""

# Simple forecasts - with / no dummy
yhat < - list()
yhat$y < - predict(ols.fit, newdata=sim.data$fore)  # no dummy
yhat$yD < - predict(olsD.fit, newdata=sim.data$fore)  # with dummy

yhat$yD.se < - sqrt(sum(olsD.fit$residuals ^ 2) / 198)
yhat$yD.up < - yhat$yD + 1.96 * yhat$yD.se
yhat$yD.low < - yhat$yD - 1.96 * yhat$yD.se

# Plot - y / yhat1 / yhat2
yhat.plot < - data.table('yhat' = rbindlist(list(data.table(sim.data$fore[, y]),
data.table(yhat$y),
data.table(yhat$yD))),
'label' = rep(c('Y', 'YHAT1', 'YHAT2'), each=100))

ggplot(yhat.plot, aes(x=rep(302: 401, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

# Plot - yhat2
yhat.plot < - data.table('yhat' = rbindlist(list(data.table(yhat$yD),
data.table(yhat$yD.up),
data.table(yhat$yD.low))),
'label' = rep(c('YHAT2', 'YHAT2_UP', 'YHAT2_LOW'),
              each=100))

ggplot(yhat.plot, aes(x=rep(302: 401, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

# Recursive
yhat$y.rec < - yhat$yD.rec < - yhat$yD.recse < - rep(0, 100)

for (i in 1:100) {
    ols.rec < - lm(y ~ x, data = sim.data$full[102:(300 + i)])
    yhat$y.rec[i] < - predict(ols.rec, newdata=sim.data$full[301 + i])
    olsD.rec < - lm(y~ D + x + D * x, data = sim.data$full[102:(300 + i)])
    yhat$yD.rec[i] < - predict(olsD.rec, newdata=sim.data$full[301 + i])
    yhat$yD.recse[i] < - sqrt(sum(olsD.rec$residuals ^ 2) / (197 + i))
}

# Plot - recursive forecasts with dummy
ggplot(yrec.plot, aes(x=rep(302: 401, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

yrec.plot < - data.table('yhat' = rbindlist(list(data.table(yhat$yD.rec),
data.table(yhat$yD.rec +1.96 * yhat$yD.recse),
data.table(yhat$yD.rec -1.96 * yhat$yD.recse))),
'label' =
rep(c('YHAT2_REC', 'YHAT2_REC_UP', 'YHAT2_REC_LOW'),
    each=100))

ggplot(yrec.plot, aes(x=rep(302: 401, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

"""

from math import sqrt

df_yhat = pd.DataFrame()
df_yhat['y'] = ols_fit.predict(df_sim_data_fore)
df_yhat['yD'] = olsD_fit.predict(df_sim_data_fore)

df_yhat['yD.se'] = sqrt(np.sum(olsD_fit.resid ** 2) / 198)
df_yhat['yD.up'] = df_yhat['yD'] + 1.96 * df_yhat['yD.se']
df_yhat['yD.low'] = df_yhat['yD'] - 1.96 * df_yhat['yD.se']

# Plot - y / yhat1 / yhat2
df_yhat_plot = pd.concat([df_sim_data_fore[['y']], df_yhat[['y']], df_yhat[['yD']]], axis=1)
df_yhat_plot.columns = ['Y', 'YHAT1', 'YHAT2']
df_yhat_plot.plot()

# Plot - yhat2
df_yhat2_plot = pd.concat([df_yhat[['yD']], df_yhat[['yD.up']], df_yhat[['yD.low']]], axis=1)
df_yhat2_plot.columns = ['YHAT2', 'YHAT2_UP', 'YHAT2_LOW']
df_yhat2_plot.plot()

df_yhat['y.rec'] = 0
df_yhat['yD.rec'] = 0
df_yhat['yD.recse'] = 0

df_rec = pd.DataFrame(index=range(1, 101), columns=['y.rec', 'yD.rec', 'yD.recse'])

# i = 0
for i in range(0, 100):
    # no dummy
    ols_rec = smf.ols(formula='y ~ x', data=df_sim_data_full[101:(301 + i)]).fit()
    df_yhat['y.rec'][301+i] = ols_rec.predict(df_sim_data_full[301 + i:(302 + i)])

    # an easy way to model structural breaks or parameter instability is by introducing dummy variables
    olsD_rec = smf.ols(formula='y ~ D + x + D * x', data = df_sim_data_full[101:(301 + i)]).fit()
    df_yhat['yD.rec'][301+i] = olsD_rec.predict(df_sim_data_full[301 + i:(302 + i)])
    df_yhat['yD.recse'][301+i] = sqrt(np.sum(olsD_rec.resid ** 2) / 197 + i)


"""

# Plot - recursive forecasts
yrec.plot < - data.table('yhat' = rbindlist(list(data.table(sim.data$fore[, y]), data.table(yhat$y.rec), data.table(yhat$yD.rec))),'label' = rep(c('Y', 'YHAT1_REC', 'YHAT2_REC'),each=100))

# RMSE & MAE
yhat$Y < - cbind(yhat$y, yhat$yD.rec)
RMSE < - sqrt(colSums((yhat$Y - sim.data$fore[, y]) ^ 2) / 100)
MAE < - colSums(abs(yhat$Y - sim.data$fore[, y])) / 100
error.mat < - rbind(RMSE, MAE)
colnames(error.mat) < - c('Simple', 'Recursive')
print(xtable(error.mat), include.rownames = T, include.colnames = T)

"""

# Plot - actual & recursive forecasts
df_plot = pd.concat([df_sim_data_fore[['y']],df_yhat['y.rec'],df_yhat['yD.rec']], axis=1)
df_plot.columns = ['Y', 'YHAT1_REC', 'YHAT2_REC']
df_plot.plot()
print(ols_rec.summary())
print(olsD_rec.summary())
plt.show()

# RMSE & MAE
df_ = pd.concat([df_yhat['y'], df_yhat['yD.rec']], axis=1)
df_RMSE = df_.apply(lambda x: sqrt(((x - df_sim_data_fore['y'].values) ** 2).sum() / 100))
df_MSE = df_.apply(lambda x: (x - df_sim_data_fore['y'].values).abs().sum() / 100)
df_error = pd.DataFrame(df_RMSE).T.append(pd.DataFrame(df_MSE).T)
df_error.columns = ['Simple', 'Recursive']
df_error = df_error.T
df_error.columns = ['RMSE', 'MAE']
df_error = df_error.T

"""
       Simple  Recursive
RMSE 47.16669   45.54340
MAE  36.14540   36.24325


eu.gdp < - fread('ex2_misspecification_gdp.csv')
gdp.fit < - lm(y~ ipr + su + sr, data = eu.gdp)


# Breusch Pagan Test
bptest(gdp.fit)
BP = 2.4511, df = 3, p-value = 0.4842

# White test
eu.gdp[, c('eps', 'eps2', 'ipr2', 'su2', 'sr2') :=list(gdp.fit$residuals, gdp.fit$residuals^2, ipr^2, su^2, sr^2)]
white.fit <- lm(eps2 ~ ipr + ipr2 + ipr * su + ipr * sr + su + su2 + su * sr + sr + sr2, data = eu.gdp)
summary(white.fit)

"""
### 2.9.1 Forecasting Euro Area GDP ###

"""

# Durbin-Watson test
dwtest(gdp.fit)  

# Breusch-Godfrey test
bgtest(y ~ ipr + su + sr, data = eu.gdp, order = 2, type = c('Chisq', 'F'))
bgtest(y ~ ipr + su + sr, data = eu.gdp, order = 3, type = c('Chisq', 'F'))

# Normality test & Histogram
hist(eu.gdp[, eps], xlab = '', ylab = '', main = '') 
jarque.bera.test(eu.gdp[, eps]) 

# Chow break point test
sctest(y ~ ipr + su + sr, data = eu.gdp, type = 'Chow', from = 0.1, to = 0.3)
sctest(y ~ ipr + su + sr, data = eu.gdp, type = 'Chow', from = 0.3, to = 0.7)
sctest(y ~ ipr + su + sr, data = eu.gdp, type = 'Chow', from = 0.7, to = 1)

fs <- Fstats(y ~ x, data = eu.gdp, from = 0.3, to = 1)
fs <- Fstats(y ~ x, data = eu.gdp, from = 0.7, to = 1)

# Bai-Perron test
breakpoints(y ~ ipr + su + sr, data = eu.gdp, h = 0.15)

"""

sim_data = r'https://storage.googleapis.com/applied-economics/ex2_misspecification_gdp.csv'
df_eu_gdp = pd.read_csv(sim_data, header=0, index_col=0, parse_dates=True).reset_index()
gdp_fit = smf.ols("y ~ ipr + su + sr", data=df_eu_gdp).fit()

# 1) Breusch-Pagan-Godfrey (BPGT) test

name = ['Lagrange multiplier statistic','p-value','f-value','f p-value']
bptest = statsmodels.stats.diagnostic.het_breuschpagan(gdp_fit.resid, gdp_fit.model.exog)
pd.DataFrame(name,bptest)

# White test
df_sq = pd.concat([gdp_fit.resid, gdp_fit.resid**2, df_eu_gdp[['ipr']]**2, df_eu_gdp[['su']]**2, df_eu_gdp[['sr']]**2], axis=1)
df_sq.columns = ['eps', 'eps2', 'ipr2', 'su2', 'sr2']
df_eu_gdp = pd.concat([df_eu_gdp, df_sq], axis=1)
white_fit = smf.ols('eps2 ~ ipr + ipr2 + ipr * su + ipr * sr + su + su2 + su * sr + sr + sr2', data=df_eu_gdp).fit()
print(white_fit.summary())

# Normality test & Histogram
df_eu_gdp[['eps']].plot()
plt.show()

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(df_sim_data_est['eps'])
lzip(name, test)


df_sq = pd.concat([df_sim_data_est['x'] **2, df_sim_data_est['eps'] **2], axis=1)
df_sq.columns = ['x2', 'eps2']

df_sim_data_est = df_sim_data_est.join(df_sq)

white_fit = smf.ols('eps2 ~ x2 + x', data=df_sim_data_est).fit()
print(white_fit.summary())

statsmodels.stats.stattools.durbin_watson(ols_fit.resid)

"""
2.45110  Lagrange multiplier statistic
0.48419                        p-value
0.79830                        f-value
0.49922                      f p-value

# Recursive estimation
gdp.rr < - recresid(y
~ ipr + su + sr, data = eu.gdp$full)
plot(gdp.rr, type='l')

# Dummy variable
gdpD.fit < - list()
gdpD.formula < - c('y ~ ipr + su + sr + Dea', 'y ~ ipr + su + sr + D2000s','y ~ ipr + su + sr + Dea + D2000s')
for (model in 1:3) {
    gdpD.fit[[model]] < - lm(gdpD.formula[model], data=eu.gdp)
    print(summary(gdpD.fit[[model]]))
}

# Forecasting
gdp.hat < - list()
gdp.fit < - lm(y
~ ipr + su + sr, data = eu.gdp[1:60])  # Model 2 - no dummy
gdp.hat$ghat < - predict(gdp.fit, newdata=eu.gdp[61:70])

gdp.fit < - lm(y~ ipr + su + sr + Dea + D2000s, data = eu.gdp[1:60])
gdp.hat$ghat3 < - predict(gdp.fit, newdata=eu.gdp[61:70])  # Model 2.3 - dummy

gdp.plot < - data.table('yhat' = rbindlist(list(data.table(eu.gdp[61:70, y]),
                                                data.table(gdp.hat$ghat),
data.table(gdp.hat$ghat3))),
'label' = rep(c('Y', 'YFOREG2_NEW', 'YFOREG2_3'),
              each=10))

ggplot(gdp.plot, aes(x=rep(1: 10, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

# RMSE & MAE
gdp.hat$Y < - cbind(gdp.hat$ghat, gdp.hat$ghat3)
RMSE < - sqrt(colSums((gdp.hat$Y - eu.gdp[61:70, y]) ^ 2) / 10)
MAE < - colSums(abs(gdp.hat$Y - eu.gdp[61:70, y])) / 10
error.mat < - rbind(RMSE, MAE)
colnames(error.mat) < - c('Model 2', 'Model 2.3')
print(xtable(error.mat), include.rownames = T, include.colnames = T)

### 2.9.2 Forecasting US GDP ###
us.gdp < - fread('ex2_misspecification_gdp_us.csv')
gdp.fit < - lm(y
~ ipr + su + sr, data = us.gdp)

# Breusch Pagan Test
bptest(gdp.fit)

# White test
us.gdp[, c('eps', 'eps2', 'ipr2', 'su2', 'sr2'): =
list(gdp.fit$residuals, gdp.fit$residuals ^ 2, ipr ^ 2, su ^ 2, sr ^ 2)]
white.fit < - lm(eps2
~ ipr + ipr2 + ipr * su + ipr * sr +
su + su2 + su * sr + sr + sr2, data = us.gdp)
summary(white.fit)

# Durbin-Watson test
dwtest(gdp.fit)

# Normality test & Histogram
hist(us.gdp[, eps], xlab = '', ylab = '', main = '')
jarque.bera.test(us.gdp[, eps])

# Chow break point test
sctest(y
~ ipr + su + sr, data = us.gdp, type = 'Chow',
from = 0.55, to = 0.6)
sctest(y
~ ipr + su + sr, data = us.gdp, type = 'Chow',
from = 0.75, to = 0.8)

fs < - Fstats(y
~ x, data = us.gdp,
from = 0.55, to = 1)
fs < - Fstats(y
~ x, data = us.gdp,
from = 0.75, to = 1)

# Bai-Perron test
breakpoints(y
~ ipr + su + sr, data = eu.gdp, h = 0.15)

# Recursive estimation
gdp.rr < - recresid(y
~ ipr + su + sr, data = us.gdp)
plot(gdp.rr, type='l')

# Dummy variable
gdpD.fit < - list()
gdpD.formula < - c('y ~ ipr + su + sr + Dfincris', 'y ~ ipr + su + sr + D2000s',
                   'y ~ ipr + su + sr + Dfincris + D2000s')
for (model in 1:3) {
    gdpD.fit[[model]] < - lm(gdpD.formula[model], data=us.gdp)
}
    summary(gdpD.fit[[1]])
summary(gdpD.fit[[2]])
summary(gdpD.fit[[3]])

## Forecasting
gdp.hat < - list()

# Model 2 - no dummy
gdp.fit < - lm(y
~ ipr + su + sr, data = us.gdp[1:104])
gdp.hat$ghat < - predict(gdp.fit, newdata=us.gdp[105:114])

# Model 2.3 - dummy
gdp.fit < - lm(y
~ ipr + su + sr + Dfincris + D2000s, data = us.gdp[1:104])
gdp.hat$ghat3 < - predict(gdp.fit, newdata=us.gdp[105:114])

gdp.plot < - data.table('yhat' = rbindlist(list(data.table(us.gdp[105:114, y]),
                                                data.table(gdp.hat$ghat),
data.table(gdp.hat$ghat3))),
'label' = rep(c('Y', 'YFOREG2_3', 'YRFOREG2_3'),
              each=10))

ggplot(gdp.plot, aes(x=rep(1: 10, 3), y = yhat, linetype = label)) +
geom_line() + xlab('') + ylab('') + theme(legend.title = element_blank())

# Recursive
for (i in 1:10) {
    ols.rec < - lm(y ~ ipr + su + sr, data = us.gdp[1:(103 + i)])
gdp.hat$rec[i] < - predict(ols.rec, newdata=us.gdp[104 + i])

olsD.rec < - lm(y
~ ipr + su + sr + Dfincris + D2000s,
data = us.gdp[1:(103 + i)])
gdp.hat$rec3[i] < - predict(olsD.rec, newdata=us.gdp[104 + i])
}

# RMSE & MAE
gdp.hat$Y < - cbind(gdp.hat$ghat, gdp.hat$ghat3)  # simple
RMSE < - sqrt(colSums((gdp.hat$Y - us.gdp[105:114, y]) ^ 2) / 10)
MAE < - colSums(abs(gdp.hat$Y - us.gdp[105:114, y])) / 10
error.mat < - rbind(RMSE, MAE)

# Recursive RMSE & MAE
gdp.hat$Yrec < - cbind(gdp.hat$rec, gdp.hat$rec3)  # recursive
RMSE < - sqrt(colSums((gdp.hat$Yrec - us.gdp[105:114, y]) ^ 2) / 10)
MAE < - colSums(abs(gdp.hat$Yrec - us.gdp[105:114, y])) / 10
error.mat < - rbind(error.mat, RMSE, MAE)
rownames(error.mat) < - c('Simple RMSE', 'Simple MAE',
                          'Recursive RMSE', 'Recursive MAE')
colnames(error.mat) < - c('Model 2', 'Model 2.3')
print(xtable(error.mat), include.rownames = T, include.colnames = T)

### 2.9.3 Default Risk ###
default.risk < - fread('default_risk.csv')
default.risk[, Date: = as.Date(Date, format='%m/%d/%Y')]
default.risk[, OAS: = OAS[2:216]]
default.risk < - default.risk[1:215]

# Dummy and interaction term
default.risk[Date >= '2008-01-01' & Date < '2010-01-01', D: = 1]
default.risk[Date < '2008-01-01' | Date >= '2010-01-01', D: = 0]
default.risk[, c('VIX.D', 'SENT.D', 'PMI.D', 'sp500.D'): =
list(VIX * D, SENT * D, PMI * D, sp500 * D)]

# Dummy
default.D < - list('M1' = 'OAS ~ VIX + D', 'M2' = 'OAS ~ SENT + D',
                                                  'M3' = 'OAS ~ PMI + D', 'M4' = 'OAS ~ sp500 + D')

# Interaction
default.I < - list('M1' = 'OAS ~ VIX + D + VIX.D',
                          'M2' = 'OAS ~ SENT + D + SENT.D',
                                 'M3' = 'OAS ~ PMI + D + PMI.D',
                                        'M4' = 'OAS ~ sp500 + D + sp500.D')

for (m in c('M1', 'M2', 'M3', 'M4')) {
    fit.D < - lm(default.D[[m]], data=default.risk)
print(summary(fit.D))
print(coeftest(fit.D, vcov = NeweyWest(fit.D, lag = 12)))
}

for (m in c('M1', 'M2', 'M3', 'M4')) {
    fit.I < - lm(default.I[[m]], data=default.risk)
print(summary(fit.I))
print(coeftest(fit.I, vcov = NeweyWest(fit.I, lag = 12)))
}
    
"""