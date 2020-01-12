# VIX Model - Beta
Note: the higher version adds more alternative data and vastly expanded the feature engineering process and some stacking is involved.

This collection of codes is an end-to-end process, which begins by fetching data via web harvesting and end with modelling and statistical analysis. In this example, we predict probabilities of price movements under a specific circumstances, in which a trader would like to enter a long call option for hedging or speculative plays.

This call option trade often requires low probabilities to break-even and a model that predicts such moves with a greater probability than the break-even probability will add tremendous values. 
