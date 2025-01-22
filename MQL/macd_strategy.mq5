#include <strategy_header.mqh>

#define LOT    0.01

input double rr_min = 3;

void OnTick()
{   
   // Check if it's within the trading session
   if (!IsTradingTime()) 
      return; // Do nothing if outside the trading session
        
   if(IsNewCandle())
   {                                                 
     if(PositionsTotal()==0)         //CountBuyPosition()==0
        {
          if(check_buy_condition(PERIOD_M15, PERIOD_H4))
          {
            double sl = NormalizeDouble(sl_buy_macd_prev_cross(Symbol(),PERIOD_M5),Digits());

            double tp = NormalizeDouble(tp_buy_macd_prev_cross(Symbol(),PERIOD_H4),Digits());
            
            if(tp!=0.0 && sl!=0.0 && (tp-Ask)/(Ask-sl)> rr_min)
              trade.Buy(LOT,NULL,Ask,sl,tp,NULL);
          }
        }
     else // Modify the existing trade's stop loss
       {
         TrailingStopLoss();
       }
 
   
   
   Comment(
           "Equity : ", AccountInfoDouble(ACCOUNT_EQUITY),"\n",
           "trade time: ",IsTradingTime(), "\n"
           ,SymbolInfoInteger(Symbol(),SYMBOL_TRADE_STOPS_LEVEL)*Point());
   }
}

