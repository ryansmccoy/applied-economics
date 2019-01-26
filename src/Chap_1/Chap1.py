from math import sqrt

import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)  # pandas
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 600)

import statsmodels.formula.api as smf

sim_data = r'https://storage.googleapis.com/applied-economics/simulated_data.csv'

df_sim_data = pd.read_csv(sim_data, header=0, index_col=0, parse_dates=True).reset_index()

df_sim_data_est = df_sim_data[102:301]
df_sim_data_fore = df_sim_data[302:401]

"""

# Y, X, and scatter plot
aes(x, y)

ggplot(sim.data$est, aes(x = 102:301, y = y)) + geom_line() +  theme_bw() + xlab('') + ylab('') + ggtitle('Y')
ggplot(sim.data$est, aes(x = 102:301, y = x)) + geom_line() +  theme_bw() + xlab('') + ylab('') + ggtitle('X')
ggplot(sim.data$est, aes(x = x, y = y)) + geom_point() +  theme_bw() + xlab('X') + ylab('Y')
"""

df_sim_data[['x', 'y']].plot.scatter(x='x', y='y')

df_sim_data[['x']].plot()
df_sim_data[['y']].plot()

## OLS
"""
ols.fit <- lm(y ~ x, data = sim.data$est)
print(xtable(ols.fit), floating = F) # LaTeX output
"""

ols_fit = smf.ols(formula='y ~ x', data=df_sim_data_est).fit()
print(ols_fit.summary())

# Forecasts
"""
yhat <- list()
yhat$y <- predict(ols.fit, newdata = sim.data$fore)
yhat$se <- sqrt(sum(ols.fit$residuals^2) / 198)
yhat$y.up <- yhat$y + 1.96 * yhat$se
yhat$y.low <- - 1.96 * yhat$se
"""

yhat = pd.DataFrame()
yhat['y_hat'] = ols_fit.predict(df_sim_data_fore)
yhat['se'] = sqrt(np.sum(ols_fit.resid ** 2) / 198)
yhat['up'] = yhat['y_hat'] + 1.96 * yhat['se']
yhat['low'] = yhat['y_hat'] - 1.96 * yhat['se']

yhat.plot()
# Plot - yhat1 / yhat1_up / yhat1_low
"""
yhat$y.rec <- yhat$y.recse <- rep(0, 100)

for (i in 1:100) {
  ols.rec <- lm(y ~ x, data = sim.data$full[102:(300 + i)])
  yhat$y.rec[i] <- predict(ols.rec, newdata = sim.data$full[301 + i])
  yhat$y.recse[i] <- sqrt(sum(ols.rec$residuals^2) / (197 + i))
}
"""

y_plot = pd.concat([df_sim_data_fore[['y']], yhat[['y_hat']]], axis=1)
y_plot.plot()

## Recursive
df_rec = pd.DataFrame(index=range(0, 400), columns=['y_rec', 'y_recse'])

for i in range(1, 100):
    ols_rec = smf.ols(formula='y ~ x', data=df_sim_data[101:(300 + i)]).fit()
    df_rec['y_rec'][i + 300] = round(float(ols_rec.predict(df_sim_data[300 + i:(301 + i)])), 6)
    df_rec['y_recse'][i + 300] = sqrt(np.sum(ols_rec.resid ** 2) / 197 + i)

# Plot - actual & recursive forecasts
df_plot = pd.concat([df_sim_data, df_rec], axis=1)
df_plot[275:399][['y', 'y_rec']].plot()
print(ols_rec.summary())
# define likelihood function

"""

Element 	        Description
Dep. Variable 	    Which variable is the response in the model
Model 	            What model you are using in the fit
Method 	            How the parameters of the model were calculated
No. Observations 	The number of observations (examples)
DF Residuals 	    Degrees of freedom of the residuals. Number of observations - number of parameters
DF Model 	        Number of parameters in the model (not including the constant term if present)

Element 	        Description
R-squared 	        The coefficient of determination. A statistical measure of how well the regression line approximates the real data points
Adj. R-squared 	    The above value adjusted based on the number of observations and the degrees-of-freedom of the residuals
F-statistic 	    A measure how significant the fit is. The mean squared error of the model divided by the mean squared error of the residuals
Prob (F-statistic) 	The probability that you would get the above statistic, given the null hypothesis that they are unrelated
Log-likelihood 	    The log of the likelihood function.
AIC 	            The Akaike Information Criterion. Adjusts the log-likelihood based on the number of observations and the complexity of the model.
BIC 	            The Bayesian Information Criterion. Similar to the AIC, but has a higher penalty for models with more parameters.

Description         The name of the term in the model
coef 	            The estimated value of the coefficient
std err 	        The basic standard error of the estimate of the coefficient. More sophisticated errors are also available.
t 	                The t-statistic value. This is a measure of how statistically significant the coefficient is.
P > |t| 	        P-value that the null-hypothesis that the coefficient = 0 is true. If it is less than the confidence level, often 0.05, 
                    it indicates that there is a statistically significant relationship between the term and the response. [95.0% Conf. Interval] 
                    The lower and upper values of the 95% confidence interval

Element 	        Description
Skewness 	        A measure of the symmetry of the data about the mean. Normally-distributed errors should be symmetrically distributed about the mean (equal amounts above and below the line).
Kurtosis 	        A measure of the shape of the distribution. Compares the amount of data close to the mean with those far away from the mean (in the tails).
Omnibus 	        D'Angostino's test. It provides a combined statistical test for the presence of skewness and kurtosis.
Prob(Omnibus) 	    The above statistic turned into a probability
Jarque-Bera 	    A different test of the skewness and kurtosis
Prob (JB) 	        The above statistic turned into a probability
Durbin-Watson 	    A test for the presence of autocorrelation (that the errors are not independent.) Often important in time-series analysis
Cond. No 	        A test for multicollinearity (if in a fit with multiple parameters, the parameters are related with each other).

"""

### 1.12.1 Forecasting Euro Area GDP ###

ex2_regress_gdp = r'https://storage.googleapis.com/applied-economics/ex2_regress_gdp.csv'
df_eu_gdp_full = pd.read_csv(ex2_regress_gdp, header=0, index_col=0, parse_dates=True).reset_index()

## Full sample - 1996Q1 to 2013Q2

gdp_formula = ['y ~ ipr + su + pr + sr',
               'y ~ ipr + su + sr',
               'y ~ ipr + su',
               'y ~ ipr + pr + sr']

fit = {}
df_fit = pd.DataFrame(index=range(0, 400), columns=['y_rec', 'y_recse'])

for i, model in enumerate(gdp_formula):
    print(model)
    fit[model] = smf.ols(formula=model, data=df_eu_gdp_full).fit()
    print(fit[model].summary())

## Estimation sample - 1996Q1 to 2006Q4
"""
eu.gdp$est <- eu.gdp$full[1:44]
eu.gdp$fore <- eu.gdp$full[45:70]

gdp.est <- list()
for (model in 1:4) {
  gdp.est[[model]] <- lm(gdp.formula[model], data = eu.gdp$est)
  summary(gdp.est[[model]])
}
## Static and recursive forecasts
gdp.fore <- list()
gdp.rec <- list()
for (model in 1:4) {
  gdp.fore[[model]] <- predict(gdp.est[[model]], newdata = eu.gdp$fore)

  gdp.rec[[model]] <- rep(0, 26)
  for (i in 1:26) {
    print(eu.gdp$full[44 + i])
    ols.rec <- lm(gdp.formula[model], data = eu.gdp$full[1:(43 + i)])
    gdp.rec[[model]][i] <- predict(ols.rec, newdata = eu.gdp$full[44 + i])
  }
}

"""
## Estimation sample - 1996Q1 to 2006Q4

df_eu_gdp_est = df_eu_gdp_full[0:44]
df_eu_gdp_fore = df_eu_gdp_full[44:70]

gdp_est = {}

for i, model in enumerate(gdp_formula):
    gdp_est[model] = smf.ols(formula=model, data=df_eu_gdp_est).fit()
    print(gdp_est[model].summary())

"""
## Static and recursive forecasts
gdp.fore < - list()
gdp.rec < - list()
for (model in 1:4) {
    gdp.fore[[model]] < - predict(gdp.est[[model]], newdata=eu.gdp$fore)

gdp.rec[[model]] < - rep(0, 26)
for (i in 1:26) {
    print(eu.gdp$full[44 + i])
ols.rec < - lm(gdp.formula[model], data=eu.gdp$full[1:(43 + i)])
gdp.rec[[model]][i] < - predict(ols.rec, newdata=eu.gdp$full[44 + i])
}
}

"""

## Static and recursive forecasts

gdp_fore = {}
gdp_rec = {}
df_gdp_rec = pd.DataFrame(index=df_eu_gdp_fore.date, columns=['{}'.format(f) for f in gdp_formula])

for i, model in enumerate(gdp_formula):
    gdp_fore[model] = gdp_est[model].predict(df_eu_gdp_fore)

    gdp_rec[model] = [0] * 26
    for i in range(0, 26):
        ols_rec = smf.ols(formula=model, data=df_eu_gdp_full[0: (44 + i)]).fit()
        df_gdp_rec['{}'.format(model)][df_eu_gdp_full.loc[[44 + i]].date] = float(ols_rec.predict(df_eu_gdp_full.loc[[44 + i]]))

# Plots - actual & forecasts
df_eu_gdp_plot = pd.concat([df_eu_gdp_full.set_index('date'), df_gdp_rec], axis=1)
pred_columns = ['y'] + ['{}'.format(f) for f in gdp_formula]
df_eu_gdp_plot[df_eu_gdp_fore.date.min(): df_eu_gdp_fore.date.max()][pred_columns].plot()

"""
# RMSE & MAE
gdp.rec$Y <- cbind(gdp.rec[[1]], gdp.rec[[2]], gdp.rec[[3]], gdp.rec[[4]])
RMSE <- sqrt(colSums((gdp.rec$Y - eu.gdp$fore[, y])^2) / 26)
MAE <- colSums(abs(gdp.rec$Y - eu.gdp$fore[, y])) / 26
error.mat <- rbind(RMSE, MAE)
"""

# RMSE & MAE
df_RMSE = df_gdp_rec.apply(lambda x: sqrt(((x - df_eu_gdp_fore['y'].values) ** 2).sum() / 26))
df_MSE = df_gdp_rec.apply(lambda x: (x - df_eu_gdp_fore['y'].values).abs().sum() / 26)

df_error = pd.concat([df_RMSE, df_MSE], axis=1)
df_error.columns = ['RMSE', 'MSE']
df_error

### 1.12.2 Forecating US GDP ###
"""
us.gdp <- list()
us.gdp$full <- fread('ex2_regress_gdp_us.csv')
us.gdp$full[, date := as.Date(date, format = '%m/%d/%Y')]
"""

ex2_regress_gdp_us = r'https://storage.googleapis.com/applied-economics/ex2_regress_gdp_us.csv'
df_us_gdp_full = pd.read_csv(ex2_regress_gdp_us, header=0, index_col=0, parse_dates=True).reset_index()

gdp_vars = ['y', 'ipr', 'sr', 'su', 'pr']

df_us_gdp_full[['date'] + gdp_vars].set_index('date').plot()

us_gdp_formula =   ['y ~ ipr + su + pr + sr','y ~ ipr + su + sr','y ~ ipr + su','y ~ ipr + pr + sr']
# Summary

gdp_fit = {}
gdp_rec = {}
df_us_gdp_rec = pd.DataFrame(index=df_us_gdp_full.date, columns=['{}'.format(f) for f in us_gdp_formula])

for i, model in enumerate(us_gdp_formula):
    gdp_fit[model] = smf.ols(formula=model, data=df_us_gdp_full).fit()
    print(gdp_fit[model].summary())

## Estimation sample - 1985Q1 to 2006Q4

df_us_gdp_est = df_us_gdp_full[0:88]
df_us_gdp_fore = df_us_gdp_full[88:116]

us_gdp_est = {}

for i, model in enumerate(us_gdp_formula):
    us_gdp_est[model] = smf.ols(formula=model, data=df_us_gdp_est).fit()
    print(us_gdp_est[model].summary())

## Static and recursive forecasts

us_gdp_fore = {}
us_gdp_rec = {}
df_us_gdp_rec = pd.DataFrame(index=df_us_gdp_fore.date, columns=['{}'.format(f) for f in us_gdp_formula])

for i, model in enumerate(us_gdp_formula):
    us_gdp_fore[model] = us_gdp_est[model].predict(df_us_gdp_fore)
    us_gdp_rec[model] = [0] * 28
    for i in range(0, 28):
        ols_rec = smf.ols(formula=model, data=df_us_gdp_full[0: (88 + i)]).fit()
        df_us_gdp_rec['{}'.format(model)][df_us_gdp_full.loc[[88 + i]].date] = float(ols_rec.predict(df_us_gdp_full.loc[[88 + i]]))

# Plots - actual & forecasts
df_us_gdp_plot = pd.concat([df_us_gdp_full.set_index('date'), df_us_gdp_rec], axis=1)
pred_columns = ['y'] + ['{}'.format(f) for f in us_gdp_formula]
df_us_gdp_plot[df_us_gdp_fore.date.min(): df_us_gdp_fore.date.max()][pred_columns].plot()

### 1.13.1 Forecasting default risk ###
"""
(OAS) Bank of America Merrill Lynch US High Yield Master II Option-Adjusted Spread, denoted OAS, monthly
(VIX) the Chicago Board Options Exchange (CBOE) Volatility Index, denoted VIX
(SENT) Surveys of Consumers, University of Michigan, consumer sentiment index, denoted SENT
(PMI the ISM Manufacturing: purcahsing managers index, PMI
(sp500) the monthly returns, in percentage points, of the S&P 500 Index

"""
ex3_regress_oas = r'https://storage.googleapis.com/applied-economics/ex3_regress_oas.csv'
df_oas = pd.read_csv(ex3_regress_oas, header=0, index_col=0, parse_dates=True).reset_index()
df_oas.set_index("Date").plot()

"""
           Date  OAS   VIX  SENT  PMI       sp500
  1: 1998-01-01 2.87 21.47 106.6 53.8  1.01501768
  2: 1998-02-01 2.83 18.55 110.4 52.9  7.04491930
"""

# shift OAS
df_oas[['OAS']] = df_oas[['OAS']].shift(-1)
df_oas.dropna(inplace=True)

"""
          Date      OAS      VIX      SENT      PMI     sp500
0   1998-01-01  2.87000 21.47000 106.60000 53.80000   1.01502
1   1998-02-01  2.83000 18.55000 110.40000 52.90000   7.04492
"""

yield_formulas = ['OAS ~ VIX', 'OAS ~ SENT', 'OAS ~ PMI', 'OAS ~ sp500', 'OAS ~ VIX + SENT + PMI + sp500']

yield_fit = {}
df_yield = pd.DataFrame(index=df_oas.Date, columns=['{}'.format(f) for f in yield_formulas])

for i, model in enumerate(yield_formulas):
    yield_fit[model] = smf.ols(formula=model, data=df_oas).fit()
    print(yield_fit[model].summary())
