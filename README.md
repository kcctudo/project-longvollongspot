# project-longvollongspot

## **Contents**
* Project description
* Data source and libraries
* Assumptions
* Output

## **Project description**
This project assessed a Long Volatility Long Spot strategy within the short dated tenor(weekly).  The strategy involves Long weekly delta hedged weekly put, long S&P 500 and long 1 week call.  We assume to trade the available weekly (MOnday, Wednesday and Friday) listed options.  We will then compare the weekly strategy to a 3 month long vol long spot strategy.  We picked weekly tenor for our strategy because of the inherited discount in the short tenor options.  The discount in the short dated option is due to increasing vol selling products that banks offered to clients.  We showed there is a superior performance in the weekly strategy vs. the 3-month strategy.  This suggested to us that there is indeed a discount in the weekly options.
## **Data source and libraries**
* Implied vol and price data are from bloomberg
* Pandas, numpy, datetime, scipy.stats and hvplot libraries
## **Assumptions**
* $10 million of Equity
* Number of Puts and Calls that will have equivalent of $10mm notion
* Daily delta hedge put options
* No hedging for call options.
## **Output**

### **Backtest Results** <br/>
Avg Cash Deployed	25,049,352 <br/> 
Annualized Return	3,087,353 <br/>
Annual Volatility	2,568,171 <br/>
Information Ratio	1.2 <br/>
Sortino Ratio	2.3 <br/>
S&P 500 IR	0.27 <br/>
LVLS 3M Tenor IR	0.31 <br/>

### **Charts** <br/>

